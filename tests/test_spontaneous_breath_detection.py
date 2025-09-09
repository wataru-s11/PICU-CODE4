from vital_reader import detect_spontaneous_breath


def make_image(with_line: bool):
    img = [[[0, 0, 0] for _ in range(20)] for _ in range(20)]
    if with_line:
        for x in range(5, 15):
            img[10][x] = [255, 255, 255]
    return img


def test_detect_spontaneous_breath():
    coords = [(5, 10, 10, 1)]
    assert detect_spontaneous_breath(make_image(True), coords)
    assert not detect_spontaneous_breath(make_image(False), coords)


