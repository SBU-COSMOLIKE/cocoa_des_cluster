"""Check survey-mask and cluster-area inputs against independent limits.

Run from activated Cocoa with covariance support:
python -m pytest projects/des_cluster/tests/covariance/test_footprint_inputs.py

Small multipole grids test how inputs reach the Gaussian, SSC and count
blocks. They do not establish the accuracy of a DES covariance forecast.
The native interface is used throughout; no forecast function is replaced
by a mock. A session-local forecast fixture runs CAMB only once.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'covariance'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
import des_cluster_covariance as galaxy
import des_cluster_joint_covariance as joint
from cosmolike_notebook_utils.covariance.geometry import (
    cap_mask, correlation_pair_area, footprint_area, footprint_mask,
    load_mask_cl, mask_area,
)
from cosmolike_notebook_utils.covariance.forecast_cluster import (
    _count_area, _count_variance, save_forecast,
)


def test_full_sky_and_measured_pair_counts(tmp_path):
    """Full-sky ordered pairs have area 4*pi times the spherical annulus.

    This tests the monopole normalization and the distinction between an
    ordered pair count and a count where each unordered pair occurs once.
    The compiled harmonic contraction must give the same area as direct
    integration over the annulus. Halving measured pair availability must
    halve the area that divides the pure-noise covariance.
    """
    area = 4.0*np.pi
    mask = np.array([4.0*np.pi])
    edges = np.array([0.0, 0.2, 0.9, np.pi])
    expected = area*2.0*np.pi*(np.cos(edges[:-1])-np.cos(edges[1:]))
    np.testing.assert_allclose(
        correlation_pair_area(area, edges, np.ones(3)), expected, rtol=2.e-14,
    )
    np.testing.assert_allclose(
        correlation_pair_area(area, edges, np.full(3, 0.5)),
        0.5*expected, rtol=2.e-14,
    )
    padded = footprint_mask(area, 8, mask)
    operators = ci.covariance_realspace_operator(
        edges_rad=edges, ell_max=8, nquad=64,
    )
    native = ci.covariance_mask_pair_area(
        edges_rad=edges, mask_cl=padded, area_sr=area,
        scalar_kernel=np.ascontiguousarray(operators[3]),
    )
    np.testing.assert_allclose(native, expected, rtol=2.e-14)
    filename = tmp_path/'mask.txt'
    np.savetxt(filename, np.column_stack([np.arange(9), padded]))
    np.testing.assert_array_equal(load_mask_cl(filename), padded)
    assert mask_area(padded) == area
    # An area rounded in a catalog header is reconciled with the monopole.
    degrees = area/(np.pi/180.0)**2
    assert footprint_area(degrees*0.9995, padded) == area
    with pytest.raises(ValueError, match='mask spectrum gives'):
        footprint_area(degrees*0.99, padded)


@pytest.mark.parametrize('spectrum', [[], [[1.0]], [0.0], [-1.0],
                                     [1.0, np.nan], [1.0, -0.1], [13.0]])
def test_invalid_mask_spectra(spectrum):
    """Malformed input must fail before native covariance calls."""
    with pytest.raises(ValueError, match='mask_cl'):
        mask_area(spectrum)
    with pytest.raises(ValueError):
        footprint_area(1000.0, spectrum)


def test_mask_cutoff_and_file_order(tmp_path):
    """A caller cannot silently discard mask modes or reorder the file."""
    spectrum = cap_mask(area_sr=0.8, ell_max=16)
    with pytest.raises(ValueError, match='raise mask_ell_max'):
        footprint_mask(0.8, 8, spectrum)
    with pytest.raises(ValueError, match='nonnegative integer'):
        footprint_mask(0.8, 16.5, spectrum)
    filename = tmp_path/'bad_mask.txt'
    np.savetxt(filename, [[0, 1.0], [2, 0.1]])
    with pytest.raises(ValueError, match='multipoles'):
        load_mask_cl(filename)
    with pytest.raises(ValueError, match='area_sr'):
        correlation_pair_area(-1.0, [0.1, 0.2], [1.0])
    for edges in ([0.1, np.nan], [-0.1, 0.2], [0.1, 4.0], [0.2, 0.1]):
        with pytest.raises(ValueError, match='edges_rad'):
            correlation_pair_area(0.8, edges, [1.0])
    for correlation in ([0.0], [np.nan], [1.0, 1.0]):
        with pytest.raises(ValueError, match='mask_pair_correlation'):
            correlation_pair_area(0.8, [0.1, 0.2], correlation)


def test_count_area_interpolation_and_zero_support():
    """Only redshifts containing selected clusters need table coverage."""
    geometry = np.ones((4, 4))
    geometry[0] = 1.0/(1.0+np.array([0.25, 0.5, 0.75, 2.0]))
    density = np.array([[1.0, 1.0, 1.0, 0.0]])
    table = np.array([[0.2, 1000.0], [0.8, 4000.0]])
    result = _count_area(table, geometry, density, 2.0)
    expected = np.array([1250.0, 2500.0, 3750.0])*(np.pi/180.0)**2
    np.testing.assert_allclose(result[:3], expected, rtol=2.e-15)
    assert result[3] == 2.0
    with pytest.raises(ValueError, match='cover the redshift'):
        _count_area(table, geometry, np.ones((1, 4)), 2.0)
    for invalid in ([[0.2, -1.0], [0.8, 2.0]],
                    [[0.8, 1.0], [0.2, 2.0]],
                    [[0.2, 1.0], [0.8, np.inf]]):
        with pytest.raises(ValueError, match='cluster_area_table'):
            _count_area(invalid, geometry, density, 2.0)


def test_distinct_windows_cauchy_bound():
    """The documented cross-window approximation must remain a covariance.

    Here a white long-mode spectrum gives less variance through a larger
    cap. Using the smaller footprint's variance for the cross term would
    imply correlation greater than one. The code must refuse this input,
    not silently return an indefinite SSC block or alter eigenvalues.
    """
    distance = np.array([0.2, 0.4])
    power = np.ones((2, 129))
    area = 0.8
    variance = ci.covariance_ssc_mask_variance(
        mask_cl=cap_mask(area, 128), area_sr=area,
        distance=distance, power=power,
    )
    np.testing.assert_array_equal(
        _count_variance(ci, np.full(2, area), area, variance,
                        distance, power, 128), variance,
    )
    smaller = _count_variance(ci, np.array([0.4, 0.0]), area, variance,
                             distance, power, 128)
    assert smaller[0] > variance[0]
    assert smaller[1] == 0.0
    with pytest.raises(ValueError, match='Cauchy bound'):
        _count_variance(ci, np.full(2, 1.6), area, variance,
                        distance, power, 128)


@pytest.fixture(scope='module')
def physical_forecast():
    """Initialize one small physical model; each test recomputes its matrices."""
    ci.set_omp_threads(n=4)
    settings = joint.configuration(
        accuracy_boost=1, ytransform=False,
        gaussian={'nonlimber': False, 'rsd': False, 'ia': 'none'},
    )
    settings.update({
        'theta_edges_arcmin': np.geomspace(2.5, 250.0, 6),
        'ell_max': 128, 'mask_ell_max': 128,
        'ng_ell': np.geomspace(2.5, 128.5, 4)-0.5,
        'angle_nquad': 64, 'halo_mass_nquad': 64, 'nwindow': 1025,
    })
    joint.initialize(interface=ci, settings=settings)
    ci.set_omp_threads(n=4)
    baseline = joint.compute(interface=ci, settings=settings)
    return settings, baseline


def test_count_area_changes_only_counts(physical_forecast, tmp_path):
    """A constant area table recovers the default; a smaller area changes N.

    At half area, mean counts and Poisson variance halve. Under the stated
    cross-window approximation count-two-point SSC halves too. Count-count
    SSC must instead match a complete forecast with that smaller area,
    because it needs that footprint's own long-mode variance.
    """
    settings, baseline = physical_forecast
    table = np.array([[0.0, settings['area_deg2']],
                      [4.0, settings['area_deg2']]])
    equal = joint.compute(interface=ci, settings={**settings, 'cluster_area_table': table})
    for key in ('gaussian', 'ssc', 'cng', 'total', 'joint_signal'):
        np.testing.assert_array_equal(equal[key], baseline[key])
    table[:, 1] *= 0.5
    reduced = joint.compute(interface=ci, settings={**settings, 'cluster_area_table': table})
    smaller = joint.compute(
        interface=ci,
        settings={**settings, 'area_deg2': settings['area_deg2']/2.0},
    )
    count = baseline['count_positions']
    two = baseline['two_point_positions']
    for key in ('gaussian', 'ssc', 'cng'):
        np.testing.assert_array_equal(
            reduced[key][np.ix_(two, two)],
            baseline[key][np.ix_(two, two)],
        )
    np.testing.assert_array_equal(reduced['mean_counts'], baseline['mean_counts']/2.0)
    np.testing.assert_array_equal(reduced['gaussian'][count, count], reduced['mean_counts'])
    np.testing.assert_array_equal(
        reduced['ssc'][np.ix_(count, two)],
        baseline['ssc'][np.ix_(count, two)]/2.0,
    )
    np.testing.assert_allclose(
        reduced['ssc'][np.ix_(count, count)],
        smaller['ssc'][np.ix_(count, count)],
        rtol=3.e-14,
    )
    production = joint.compute(
        interface=ci,
        settings={**settings, 'cluster_area_table': table},
        backend=ci.covariance,
    )
    for key in ('gaussian', 'ssc', 'cng', 'total'):
        np.testing.assert_array_equal(reduced[key], production[key])
    filename = tmp_path/'area_forecast.npz'
    save_forecast(result=reduced, filename=filename)
    with np.load(filename, allow_pickle=False) as saved:
        assert 'cluster_area_table' in str(saved['settings_json'])
        assert 'cross-spectrum of distinct' in str(saved['settings_json'])


def test_joint_mask_and_measured_pairs(physical_forecast):
    """Passing the same cap is exact; measured pair counts affect only G."""
    settings, baseline = physical_forecast
    area = settings['area_deg2']*(np.pi/180.0)**2
    mask = cap_mask(area, settings['mask_ell_max'])
    same = joint.compute(interface=ci, settings={**settings, 'mask_cl': mask})
    for key in ('gaussian', 'ssc', 'cng', 'total'):
        np.testing.assert_array_equal(same[key], baseline[key])
    measured = {**settings, 'mask_cl': mask, 'mask_pair_correlation': np.full(5, 0.5)}
    result = joint.compute(interface=ci, settings=measured)
    np.testing.assert_array_equal(result['ssc'], baseline['ssc'])
    np.testing.assert_array_equal(result['cng'], baseline['cng'])
    assert not np.array_equal(result['gaussian'], baseline['gaussian'])
    expected = correlation_pair_area(
        area_sr=area,
        edges_rad=settings['theta_edges_arcmin']*np.pi/10800.0,
        correlation=np.full(5, 0.5),
    )
    np.testing.assert_array_equal(result['pair_area_sr2'], expected)
    # Joint use of both PRs: matching count areas retain the supplied mask.
    measured['cluster_area_table'] = [[0.0, settings['area_deg2']], [4.0, settings['area_deg2']]]
    ci.set_omp_threads(n=1)
    production = joint.compute(interface=ci, settings=measured, backend=ci.covariance)
    ci.set_omp_threads(n=4)
    for key in ('gaussian', 'ssc', 'cng', 'total'):
        np.testing.assert_array_equal(production[key], result[key])


def test_galaxy_mask_routes(physical_forecast):
    """Check ordinary forecasts too, including the direct production binding."""
    settings, _ = physical_forecast
    rows = np.array([[0, 6, 6], [1, 6, 6], [2, 0, 6], [3, 0, 0]], dtype=np.int32)
    baseline = galaxy.compute(interface=ci, settings=settings, rows=rows)
    area = settings['area_deg2']*(np.pi/180.0)**2
    measured = {**settings, 'mask_cl': cap_mask(area, 128),
                'mask_pair_correlation': np.full(5, 0.75)}
    result = galaxy.compute(interface=ci, settings=measured, rows=rows)
    for key in ('ssc', 'cng'):
        np.testing.assert_array_equal(result[key], baseline[key])
    assert not np.array_equal(result['gaussian'], baseline['gaussian'])
    production = galaxy.compute(interface=ci, settings=measured, rows=rows, backend=ci.covariance)
    for key in ('gaussian', 'ssc', 'cng', 'total'):
        np.testing.assert_array_equal(production[key], result[key])
    with pytest.raises(ValueError, match='real-space angular bins'):
        galaxy.compute(interface=ci, settings=measured, space='fourier')


def test_disjoint_mask_changes_ssc(physical_forecast):
    """Two antipodal caps have the same area but different long-mode power.

    Odd multipoles cancel between the caps; even coefficients add before
    squaring. This supplies a realizable binary mask independent of the
    single-cap default. Equal count and two-point areas must reuse this
    mask, rather than reconstructing a cap from its monopole.
    """
    settings, baseline = physical_forecast
    area = settings['area_deg2']*(np.pi/180.0)**2
    ell = np.arange(settings['mask_ell_max']+1)
    mask = 2.0*(1.0+(-1.0)**ell)*cap_mask(area/2.0, settings['mask_ell_max'])
    changed = {**settings, 'mask_cl': mask}
    result = joint.compute(interface=ci, settings=changed)
    assert not np.array_equal(result['ssc'], baseline['ssc'])
    np.testing.assert_array_equal(result['cng'], baseline['cng'])
    changed['cluster_area_table'] = [[0.0, settings['area_deg2']],
                                     [4.0, settings['area_deg2']]]
    with_table = joint.compute(interface=ci, settings=changed)
    np.testing.assert_array_equal(with_table['ssc'], result['ssc'])


def test_yaml_geometry_inputs(tmp_path):
    """The CLI keeps geometry arrays out of the accuracy-boost resolver."""
    from cobaya.yaml import yaml_dump, yaml_load_file
    from cosmolike_notebook_utils.covariance.command_line import load_run_configuration

    info = yaml_load_file(
        file_name=str(project/'EXAMPLE_EVALUATE_COVARIANCE.yaml'),
    )
    info['theory']['camb']['path'] = str(project.parents[1]/'external_modules/code/CAMB')
    info['covariance']['mask_cl'] = [1.0, 0.2, 0.1]
    info['covariance']['mask_pair_correlation'] = [0.9, 0.8]
    info['covariance']['cluster_area_table'] = [[0.0, 1000.0], [1.0, 900.0]]
    filename = tmp_path/'survey.yaml'
    filename.write_text(yaml_dump(info=info))
    settings, _ = load_run_configuration(filename=filename, survey=joint, joint=True)
    for key in ('mask_cl', 'mask_pair_correlation', 'cluster_area_table'):
        assert settings[key] == info['covariance'][key]
    with pytest.raises(ValueError, match='requires the joint cluster'):
        load_run_configuration(filename=filename, survey=galaxy)
    del info['covariance']['cluster_area_table']
    info['covariance']['space'] = 'fourier'
    filename.write_text(yaml_dump(info=info))
    with pytest.raises(ValueError, match='real-space angular bins'):
        load_run_configuration(filename=filename, survey=galaxy)
