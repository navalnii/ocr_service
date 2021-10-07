## OCR service 
The service able to extract text from pdf, png, jpeg, jpg format file. To run it on 
- windows
install tesseract for windows to lib directory. https://github.com/UB-Mannheim/tesseract/wiki
(poopler for convert_from_path (pdf2image package) will install automatic in lib directory)
- linux and macos not nesessary to install tesseract. It should install via run poerty

run: 
cd ocr_service
python main.py --input=./test/1.pdf --output=./test/1.txt

The input file have to be {pdf, png, jpeg, jpg} only format. Output file have to be only txt format

Future:
1) Detect vertical text in image
2) Do ocr with tesserocr. It should be faster than tesseract
3) Convert code into async mode. It will be faster with pdf2image and tesserocr

