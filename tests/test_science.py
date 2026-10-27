"""Known-answer tests for leakage, timestamp alignment, and scientific arithmetic."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_dataset import hourly_flux, fill_short_gaps, sample_indices
from baselines import scores, fit_diurnal
from data_validation import validate_omni, validate_goes
from scipy.io import netcdf_file


class ScienceTests(unittest.TestCase):
    def test_flux_order_flags_floor_missing(self):
        f = np.r_[np.ones(6), np.full(6, 100.), np.full(12, .1), np.full(12, -99999.), np.arange(1., 13.)]
        q = np.zeros(48)
        q[6:12] = 1
        result, count = hourly_flux(f, q)
        np.testing.assert_allclose(result[[0, 1, 3]], [0, 0, np.log10(6.5)])
        self.assertTrue(np.isnan(result[2]))
        self.assertEqual(count, 1)

    def test_insufficient_samples_and_missing_timestamp(self):
        self.assertTrue(np.isnan(hourly_flux(np.ones(12), np.r_[np.zeros(5), np.ones(7)])[0][0]))
        result, _ = hourly_flux(np.ones(24), np.zeros(24), np.r_[np.zeros(12, int), np.full(12, 2)], 3)
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_array_equal(result[[0, 2]], [0, 0])

    def test_complete_short_bounded_gaps(self):
        v = np.arange(20., dtype=float)[:, None]
        v[[0, 2, 3, 4, 8, 9, 10, 11, 19]] = np.nan
        filled, available = fill_short_gaps(v, np.zeros(20))
        np.testing.assert_allclose(filled[2:5, 0], [2, 3, 4])
        np.testing.assert_array_equal(available[2:5, 0], [5, 5, 5])
        self.assertTrue(np.isnan(filled[[0, 8, 9, 10, 11, 19]]).all())

    def test_no_cross_period_interpolation(self):
        raw = np.array([1., np.nan, 3.])[:, None]
        self.assertTrue(np.isnan(fill_short_gaps(raw, np.array([0, 0, 1]))[0][1, 0]))

    def test_splits_observed_labels_future_perturbation(self):
        time = np.arange(np.datetime64('2015-12-20T00', 'h'), np.datetime64('2017-01-10T00', 'h'))
        raw = np.tile(np.arange(len(time), dtype=float)[:, None], (1, 4))
        periods = time.astype('datetime64[Y]').astype(int)
        raw[100, 1] = np.nan
        filled, available = fill_short_gaps(raw, periods)
        idx, _ = sample_indices(raw, filled, available, time)
        self.assertNotIn(100, idx['train'])
        altered = raw.copy()
        altered[101:] += 10000
        filled2, _ = fill_short_gaps(altered, periods)
        for i in idx['train'][idx['train'] < 100]:
            np.testing.assert_equal(filled[i - 71:i + 1], filled2[i - 71:i + 1])
        for name, end in [('train', '2016'), ('val', '2017'), ('test', '2018')]:
            self.assertTrue(((time[idx[name]] + np.timedelta64(13, 'h')) < np.datetime64(end)).all())
        first_val = idx['val'][0]
        self.assertLess(time[first_val - 71], np.datetime64('2016'))
        raw[first_val + 12, 0] = np.nan
        filled, available = fill_short_gaps(raw, periods)
        revised, _ = sample_indices(raw, filled, available, time)
        self.assertNotIn(first_val, revised['val'])

    def test_diurnal_fit_uses_only_training(self):
        raw = np.column_stack([np.sin(np.arange(1000) / 10)] * 4)
        lt, idx = np.arange(1000) % 24, np.arange(100, 600)
        first = fit_diurnal(raw, lt, idx)
        raw[612:] += 1000
        second = fit_diurnal(raw, lt, idx)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue((first[1] > 0).all())

    def test_metric_known_answers(self):
        observed = np.array([0., 1., 2.])
        self.assertEqual(scores(observed, observed)['pe'], 1.)
        self.assertEqual(scores(observed, np.ones(3))['pe'], 0.)
        self.assertAlmostEqual(scores(observed, np.ones(3))['rmse'], np.sqrt(2 / 3))
        with self.assertRaises(ValueError):
            scores(np.ones(3), np.ones(3))
        with self.assertRaises(ValueError):
            scores(observed, np.array([0, np.nan, 2]))

    def test_omni_leap_calendar(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'year.dat'
            values = np.ones((8784, 55))
            values[:, 0], values[:, 1], values[:, 2] = 2012, np.arange(8784) // 24 + 1, np.arange(8784) % 24
            np.savetxt(p, values, fmt='%.1f')
            self.assertEqual(validate_omni(p, 2012)['rows'], 8784)
            values[100, 2] = 0
            np.savetxt(p, values, fmt='%.1f')
            with self.assertRaises(ValueError):
                validate_omni(p, 2012)

    def test_goes_gaps_explicit_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'month.nc'
            for duplicate in [False, True]:
                with netcdf_file(p, 'w') as data:
                    data.createDimension('record', 12)
                    t = data.createVariable('time_tag', 'd', ('record',))
                    t.units = 'milliseconds since 1970-01-01 00:00:00'
                    ticks = np.datetime64('2012-03-01', 'ms').astype('int64') + np.arange(12) * 300000
                    if duplicate:
                        ticks[1] = ticks[0]
                    t[:] = ticks
                    for sensor in ['E2E', 'E2W']:
                        data.createVariable(sensor + '_COR_FLUX', 'd', ('record',))[:] = np.ones(12)
                        data.createVariable(sensor + '_QUAL_FLAG', 'i', ('record',))[:] = np.zeros(12, int)
                if duplicate:
                    with self.assertRaises(ValueError):
                        validate_goes(p, 2012, 3)
                else:
                    self.assertEqual(validate_goes(p, 2012, 3)['absent_grid_records'], 31 * 288 - 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
