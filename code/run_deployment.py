import os

os.system("python shortcut_detector_deployment.py /mnt/SSD/covidx/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 covidx")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 uw")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 sp")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 midrc")
os.system("python shortcut_detector_deployment.py /mnt/SSD/real_fake_cxr/ ../weights/shortcut_detective/MIMIC_ADAC/ densenet 5 roentgen")


os.system("python shortcut_detector_deployment.py /mnt/SSD/covidx/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 covidx")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 uw")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 sp")
os.system("python shortcut_detector_deployment.py /mnt/SSD/covid_png/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 midrc")
os.system("python shortcut_detector_deployment.py /mnt/SSD/real_fake_cxr/ ../weights/shortcut_detective/MIMIC_ADAS/ densenet 5 roentgen")
