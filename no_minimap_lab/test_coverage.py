import unittest
from .coverage_trial import build_plan, is_travel_portal, rope_takeoff
from types import SimpleNamespace


class RouteCoverage(unittest.TestCase):
    def test_side_rope_approach_stays_on_platform_until_jump(self):
        node=SimpleNamespace(x_min=235,x_max=364)
        self.assertEqual(rope_takeoff(node,224),243)
        self.assertEqual(rope_takeoff(node,322),322)

    def test_spawn_markers_do_not_block_rope_but_real_portals_do(self):
        self.assertFalse(is_travel_portal(dict(pt=0,tm=999999999)))
        self.assertTrue(is_travel_portal(dict(pt=10,tm=101000000)))
        self.assertTrue(is_travel_portal(dict(pt=2,tm=101000001)))

    def test_all_physical_ropes_have_connected_explicit_traversals(self):
        plan=build_plan()
        self.assertEqual([s['ladder']['id'] for s in plan['steps']],list(range(1,38)))
        current=1
        for s in plan['steps']:
            for e in s['approach']:
                self.assertEqual(e['from_id'],current)
                self.assertNotIn('TELEPORT',e['action'])
                self.assertNotEqual(e['action'],'PORTAL')
                current=e['to_id']
            rope=s['ladder']
            self.assertEqual(current,rope['bottom_platform_id'] or rope['top_platform_id'])
            if s['traversal']:
                self.assertEqual(s['traversal']['ladder_id'],rope['id'])
                current=s['traversal']['to_id']
            else:
                self.assertIsNone(rope['bottom_platform_id'])
        for e in plan['finish']:
            self.assertEqual(e['from_id'],current)
            current=e['to_id']
        self.assertEqual(current,87)


if __name__=='__main__':
    unittest.main()
