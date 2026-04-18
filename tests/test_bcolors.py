from iara.utils import Bcolors


def test_all_codes_are_strings():
    for attr in ("HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING", "FAIL", "ENDC", "BOLD", "UNDERLINE"):
        assert isinstance(getattr(Bcolors, attr), str)


def test_all_codes_start_with_escape():
    for attr in ("HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING", "FAIL", "ENDC", "BOLD", "UNDERLINE"):
        assert getattr(Bcolors, attr).startswith("\033[")


def test_endc_resets():
    assert Bcolors.ENDC == "\033[0m"


def test_codes_are_distinct():
    codes = [Bcolors.HEADER, Bcolors.OKBLUE, Bcolors.OKCYAN, Bcolors.OKGREEN,
             Bcolors.WARNING, Bcolors.FAIL, Bcolors.BOLD, Bcolors.UNDERLINE]
    assert len(codes) == len(set(codes))
