import unittest
from .map_preview import projected_player


class MapPredictionTests(unittest.TestCase):
    def test_advances_between_results_without_mutating_measurement(self):
        record = dict(timestamp=10., player_state=dict(world=[100.,200.], velocity=[120.,-20.], observation_age=0))
        self.assertEqual(projected_player(record,10.), [100.,200.])
        later = projected_player(record,10.1)
        self.assertAlmostEqual(later[0],112.)
        self.assertAlmostEqual(later[1],198.)
        self.assertEqual(record['player_state']['world'],[100.,200.])
        self.assertIsNone(projected_player(record,10.251))

    def test_already_predicted_observation_does_not_get_new_lifetime(self):
        record = dict(timestamp=10., player_state=dict(world=[1,2],velocity=[100,0],observation_age=.20))
        self.assertIsNotNone(projected_player(record,10.04))
        self.assertIsNone(projected_player(record,10.06))
        self.assertIsNone(projected_player(record,9.99))


if __name__ == '__main__':
    unittest.main()
