from projects.data_preparation.data_generator import DataGenerator
from projects.unit_test.unitTest import UnitTest

class UnitTestDataGenerator(UnitTest):
    def __init__(self, module, fn_list):
        super(UnitTest, self).__init__()


def main():
    dg = DataGenerator(output='/tmp/dots_test.pkl')
    ut = UnitTest(module=dg, fn_list=['generate_data'])
    ut.run_unit_tests()

if __name__ == '__main__':
    main()


