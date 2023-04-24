import os

os.system("python test_covid_model.py /mnt/SSD/covid_png/ HF HF")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ HF UW")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ HF BIMCV")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ HF MIDRC")

os.system("python test_covid_model.py /mnt/SSD/covidx/ covidx covidx")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ covidx UW")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ covidx BIMCV")
os.system("python test_covid_model.py /mnt/SSD/covid_png/ covidx MIDRC")

