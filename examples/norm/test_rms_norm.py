import tilelang.testing
import rms_norm


@tilelang.testing.pytest.mark.xfail
def test_rms_norm():
    rms_norm.test_rms_norm()


if __name__ == "__main__":
    tilelang.testing.main()
