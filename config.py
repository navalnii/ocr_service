import os
import sys 


def get_poopler_win32():
    import requests
    import py7zr

    if Path(Path.cwd().joinpath('lib/poppler-0.68.0')).is_dir():
        for path, directories, files in os.walk(str(Path.cwd())):
            if 'bin' in directories:
                poopler_dir = Path(path).joinpath('bin')
                return poopler_dir
                
    else:
        URL = 'https://blog.alivate.com.au/wp-content/uploads/2018/10/poppler-0.68.0_x86.7z'
        headers={'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}

        filename = os.path.basename(URL)
        response = requests.get(URL,  headers=headers)


        if response.status_code == 200:
            with open(filename, 'wb') as out:
                out.write(response.content)    

            with py7zr.SevenZipFile(filename, mode='r') as z:
                z.extractall('lib')
                
            for path, directories, files in os.walk(str(Path.cwd())):
                if 'bin' in directories:
                    poopler_dir = Path(path).joinpath('bin')
                    return poopler_dir


if sys.platform == 'win32':
    from pathlib import Path

    poopler_dir = get_poopler_win32()
    pytesseract_path = Path.cwd().joinpath('lib/Tesseract-OCR/tesseract.exe')

    
elif sys.platform == 'linux':

    from subprocess import STDOUT, check_call
    check_call(['apt-get', 'install', '-y', 'python-distutils-extra tesseract-ocr tesseract-ocr-eng \
                        libopencv-dev libtesseract-dev libleptonica-dev python-all-dev swig libcv-dev python-opencv \
                        python-numpy python-setuptools build-essential subversion tesseract-ocr-eng tesseract-ocr-dev\
                        libleptonica-dev python-all-dev swig libcv-dev'],
                stdout=open(os.devnull,'wb'), stderr=STDOUT) 
    
    check_call(['apt-get', 'install', '-y', 'poppler-utils'],
                stdout=open(os.devnull,'wb'), stderr=STDOUT) 

elif sys.platform == 'darwin':

    os.system('ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" 2> /dev/null')
    os.system('brew install poppler')
    os.system('brew install tesseract')



