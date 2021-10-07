import os
import sys
import click
import shutil
from pathlib import Path
import logging
import pytesseract
from pdf2image import convert_from_path
import time
import pickle
import cv2
from PIL import Image
import os
import numpy as np
import re
import config
from spellchecker import SpellChecker

if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = str(config.pytesseract_path)

supported_format = {'pdf', 'jpg', 'jpeg', 'png'}


logger = logging.getLogger('ocr_logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)


@click.command()
@click.option('--input', prompt='Your input file is: ', help='Input file. Supported pdf, png, jpg format')
@click.option('--output', help='Output file. Extracted text from input.')
def extract_text(input, output):
    """
    python main.py --input=./{input_name} --output=./test/{output_name.txt} --verbose
    """

    input_path = Path(input)
    input_format = input.split('/')[-1].split('.')[-1]
    output_path = Path(output)
    output_format = output.split('/')[-1].split('.')[-1]

    if input_path.is_file() and input_format in supported_format:
        logger.info(f'The input file {input_path} has been read')
        
        # Create txt output text, according output
        if output_format == 'txt':
            with open(output_path, 'w') as output_file:
                output_file.write('')

            ocr_s = ocrService()
            ocr_s.input_file(input)
            text = ocr_s.correct_spelling()

            with open(output_path, 'w', encoding='utf8') as output_file:
                output_file.write(text)
            logger.info('Correct spelled text from input has been saved')

        else:
            logger.warning(f'Incorrect output file!')
            
    else:
        logger.warning(
            f'The input file {input_path} does not exists or not supported format!')


class ocrService:
    def __init__(self):
        self.in_name = ''
        self.in_format = ''
        self.path = ''
        self.output_imgs = []
        self.output_imgs_path = []
        self.roi_path = []
        self.roi_name = []
        self.out_text = ''


    def input_file(self, input):
        self.path = Path.cwd().joinpath(input)
        self.in_name = input.split('/')[-1].split('.')[-2]
        self.in_format = input.split('/')[-1].split('.')[-1]

        if Path(self.path).is_file():

            if self.in_format == 'pdf':
                logger.info('Begining process pdf file')
                self.pdf_format_process()

            elif self.in_format in supported_format:
                logger.info(f'Begining process {self.in_format} file')
                self.im_format_process()


    def pdf_format_process(self):

        if self.pdf_to_image():

            for ind, im_path in enumerate(self.output_imgs_path):
                img = cv2.imread(str(im_path))
                rotated, res = self.get_straighten_image(img)
                clean = self.remove_horizontal_lines(rotated)
                self.split_to_contour_n_save(clean, ind, im_path)

            self.do_ocr()
            self.remove_temp_image()
            
        else:
            logger.warning(f'The pdf file {self.in_name} not found')


    def im_format_process(self):

        img = cv2.imread(str(self.path))
        rotated, res = self.get_straighten_image(img)
        clean = self.remove_horizontal_lines(rotated)
        self.split_to_contour_n_save(clean)
        self.do_ocr()
        self.remove_temp_image()
        

    def pdf_to_image(self):
        try:
            if sys.platform == 'win32':
                pages = convert_from_path(self.path, dpi=350,
                                          thread_count=5,
                                          poppler_path=str(config.poopler_dir))
            else:
                pages = convert_from_path(self.path, dpi=350,
                                          thread_count=5)
            # Create directory for each pdf with pdf-name

            filepath = Path(self.path).parent.joinpath(f'./{self.in_name}')
            Path(filepath).mkdir(parents=True, exist_ok=True)

            for i, page in enumerate(pages):

                # Save each img name and its path
                self.output_imgs.append(f'{i}.jpeg')
                self.output_imgs_path.append(
                    Path.cwd().joinpath(filepath, f"./{i}.jpeg"))

                # Save each page of pdf, in pdf-name dir
                page.save(f'{filepath}/{i}.jpeg', 'JPEG')

            return True

        except:
            logger.exception('message')
            return False

# --------------------------------------------------------------------------------
# --------------------------Preprocessing image-----------------------------------
# --------------------------------------------------------------------------------


    def get_skew_angle(self, img):

        # Prep image, copy, convert to gray scale, blur, and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 5), 0)
        thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find all contours
        contours, hierarchy = cv2.findContours(
            dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            angle = cv2.minAreaRect(c)[-1]
            area = cv2.contourArea(c)

            if abs(angle) < 45 and abs(angle) > 0.001:

                logger.info(f'Get angle {angle}')
                return angle*1.0, img

            elif abs(angle) > 45 and abs(angle) < 90:

                angle -= 90
                logger.info(f'Get angle {angle}')
                return angle*1.0, img

        return None


    def get_straighten_image(self, img_):

        angle, res = self.get_skew_angle(img_)

        if angle:
            (h, w) = img_.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_, rotation_matrix, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            logger.info(f'The image has been rotated')
            return rotated, res

        else:
            return img_, None


    def remove_horizontal_lines(self, img_):

        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        # (1) Create long line kernel, and do morph-close-op
        kernel = np.ones((1, 25), np.uint8)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

        # (2) Invert the morphed image, and add to the source image:
        dst = cv2.add(gray, (255-morphed))
        logger.info(f'In the image removed any horizontal lines')

        return dst


    def split_to_contour_n_save(self, clean, index_img=None, output_ing_path=None):

        if self.in_format == 'pdf':

            image_name = str(self.output_imgs[index_img]).split('.')[0]
            contour_path = Path(output_ing_path).parent.joinpath(
                f'./{image_name}')
            Path(contour_path).mkdir(parents=True, exist_ok=True)

        elif self.in_format in {'png', 'jpg', 'jpeg'}:

            image_name = self.in_name
            contour_path = self.path.parent.joinpath(f'./{image_name}')
            Path(contour_path).mkdir(parents=True, exist_ok=True)

        blur = cv2.GaussianBlur(clean, (9, 9), 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilate = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours, highlight text areas, and extract ROIs
        cnts, _ = cv2.findContours(
            dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c_indx, c in enumerate(cnts):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)

            if y < 100 or area > 10000:
                image = cv2.rectangle(
                    clean, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
                img_rgb = cv2.cvtColor(
                    image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)

                cnt_path = contour_path.joinpath(f'./{c_indx}.jpg')
                
                self.roi_path.append(cnt_path)
                self.roi_name.append(f'{c_indx}.jpg')
                
                cv2.imwrite(str(cnt_path), img_rgb)
                logger.info(
                    f'The cnt {c_indx}.jpg with coords:{y,y + h, x,x + w} saved')


    def remove_temp_image(self):
        try:
            if self.in_format == 'pdf':
                shutil.rmtree(self.roi_path[0].parent.parent)
            else:
                shutil.rmtree(self.roi_path[0].parent)
        except:
            logger.exception('message')

# --------------------------------------------------------------------------------
# --------------------------------------NLP---------------------------------------
# --------------------------------------------------------------------------------


    def correct_spelling(self):

        spell = SpellChecker(distance=1)
        
        logger.info('Start correct spelling')
        return ' '.join([spell.correction(word) for word in self.out_text.split()])


    def do_ocr(self):

        for roi_path in self.roi_path:
            ocr_text = Image.open(str(roi_path))
            logger.info(
                f'The contour {str(roi_path).split("/")[-1]} processed with OCR')
            self.out_text += pytesseract.image_to_string(
                ocr_text, lang="eng") + '\n'


if __name__ == '__main__':
    extract_text()