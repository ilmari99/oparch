def add_local_oparch_path():
    """ To use he local version of oparch, add the local path to the system path.
    Adds the location where the script is executed to the system path.
    """
    import sys
    from pathlib import Path
    path_root = Path(__file__).parents[1]
    sys.path.insert(0, str(path_root))
if __name__=="__main__":
    # If run as main, add local path to system path
    add_local_oparch_path()
import unittest
import oparch

class test_configure(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_set_misc(self):
        oparch.configurations.set_default_misc(validation_split=0.69,
        batch_size=123,
        epochs=1,
        samples=123,
        verbose=0,
        learning_metric="NEG_ACCURACY",
        new_value=0.001,)
        default_dict = oparch.get_default_misc()
        self.assertEqual(default_dict["validation_split"],0.69)
        self.assertEqual(default_dict["batch_size"],123)
        self.assertEqual(default_dict["epochs"],1)
        self.assertEqual(default_dict["samples"],123)
        self.assertEqual(default_dict["verbose"],0)
        self.assertEqual(default_dict["learning_metric"],"NEG_ACCURACY")
        self.assertEqual(default_dict["new_value"],0.001)
        self.assertEqual(oparch.get_default_misc("validation_split"),0.69)
        self.assertEqual(oparch.get_default_misc("batch_size"),123)
        self.assertEqual(oparch.get_default_misc("epochs"),1)
        self.assertEqual(oparch.get_default_misc("samples"),123)
        self.assertEqual(oparch.get_default_misc("verbose"),0)
        self.assertEqual(oparch.get_default_misc("learning_metric"),"NEG_ACCURACY")
        self.assertEqual(oparch.get_default_misc("new_value"),0.001)

    def test_set_intervals(self):
        oparch.set_default_intervals(
        learning_rate=[0.001,0.01],
        decay=[0.001,0.01],
        momentum=[0.001,0.01],
        rho=[0.001,0.01],
        new_value=[0.001,0.01],
        )
        default_dict = oparch.get_default_interval()
        self.assertEqual(default_dict["learning_rate"],[0.001,0.01])
        self.assertEqual(default_dict["decay"],[0.001,0.01])
        self.assertEqual(default_dict["momentum"],[0.001,0.01])
        self.assertEqual(default_dict["rho"],[0.001,0.01])
        self.assertEqual(default_dict["new_value"],[0.001,0.01])
        self.assertEqual(default_dict["activation"],list(oparch.configurations.ACTIVATION_FUNCTIONS.keys()))
        self.assertEqual(oparch.get_default_interval("learning_rate"),[0.001,0.01])
        self.assertEqual(oparch.get_default_interval("decay"),[0.001,0.01])
        self.assertEqual(oparch.get_default_interval("momentum"),[0.001,0.01])
        self.assertEqual(oparch.get_default_interval("rho"),[0.001,0.01])
        self.assertEqual(oparch.get_default_interval("new_value"),[0.001,0.01])

    def test_dicts_raise_key_error(self):
        with self.assertRaises(KeyError):
            oparch.get_default_misc("not_a_key")
        with self.assertRaises(KeyError):
            oparch.get_default_interval("not_a_key")

    
    

if __name__ == "__main__":
    unittest.main()

    