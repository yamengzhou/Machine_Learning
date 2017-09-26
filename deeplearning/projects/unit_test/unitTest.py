
class UnitTest(object):
    def __init__(self, module, fn_list):
        super(UnitTest, self).__init__()
        self.successes = []
        self.failures = []
        self.pass_ratio = []
        self.fn_list = fn_list
        self.module = module

    def run_unit_tests(self):
        for name in self.fn_list:
            method = getattr(self.module, name)
            method()