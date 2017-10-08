import pytest


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)


def pytest_addoption(parser):
    parser.addoption("--style_image", action="store",
                     type=str, required=True,
                     help="Source Image Path")
    parser.addoption("--content_image", action="store",
                     type=str, required=True,
                     help="Content Image Path")


@pytest.fixture
def style_image(request):
    return request.config.getoption("--style_image")


@pytest.fixture
def content_image(request):
    return request.config.getoption("--content_image")
