"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import *







def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    def nothing(gamma):
        gamma = cv2.getTrackbarPos('gamma', 'image')
        if (gamma != 0):
            img = pow(imgOrig, (1.0 / (gamma / 100)));
        cv2.imshow('image', img)
        k = cv2.waitKey(1)

    pass
    # Create a black image, a window
    if(rep==LOAD_GRAY_SCALE):
        img = imReadAndConvert(img_path,rep)
        imgOrig = img
    else:
        img=cv2.imread(img_path)/255
        imgOrig = img
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('gamma', 'image', 1, 200, nothing)
    # Show some stuff
    nothing(0)

    # Wait until user press some key
    cv2.waitKey()
    pass


def main():
    gammaDisplay('sample_image.jpg', 2)


if __name__ == '__main__':
    main()
