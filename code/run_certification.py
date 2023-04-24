import os

os.system("python shortcut_detector_certification.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 sharpness positive 0")
os.system("python shortcut_detector_certification.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 contrast positive 0")

