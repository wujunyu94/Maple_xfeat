from pathlib import Path
from contextlib import nullcontext
import uuid
import unittest
from unittest.mock import patch, Mock
from . import trial_runner


class TrialRunnerTests(unittest.TestCase):
    def test_command_matches_original_cli(self):
        args = trial_runner.command(Path('output/test'))
        self.assertEqual(args[1:7], ['-m','no_minimap_lab.coverage_trial','--backend','xfeat','--rounds','1'])
        self.assertEqual(args[-2:], ['--output',str(Path('output/test'))])

    def test_launch_stop_and_log_without_real_input(self):
        with nullcontext(trial_runner.ROOT/'no_minimap_lab'/'output'/('trial_launcher_test_'+uuid.uuid4().hex)) as root:
            root.mkdir()
            state = root/'control';state.mkdir()
            process = Mock();process.poll.return_value = None
            with patch.object(trial_runner,'ROOT',root), patch.object(trial_runner,'STATE',state), \
                 patch.object(trial_runner,'service_ready',return_value=True), \
                 patch.object(trial_runner.psutil,'process_iter',return_value=[]), \
                 patch.object(trial_runner.subprocess,'Popen',return_value=process) as popen:
                runner=trial_runner.TrialRunner();output=runner.start()
                self.assertTrue((output/'console.log').exists())
                self.assertEqual(popen.call_args.args[0], trial_runner.command(output))
                runner.stop()
                self.assertTrue((state/'stop_navigation').exists())
                with self.assertRaises(RuntimeError):runner.start()


if __name__ == '__main__':
    unittest.main()
