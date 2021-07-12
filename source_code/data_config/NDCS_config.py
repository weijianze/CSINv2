# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        ####################### testing dataset #########################
        # NDCrossSensor
        self._root_path_gallery = ['../../Data01_recognition/NDCS/NormIm/']
        self._gallery_list = ['../../Data01_recognition/NDCS/LG2200_test_filtered.txt']

        self._root_path_probe = ['../../Data01_recognition/NDCS/NormIm/']
        self._probe_list = ['../../Data01_recognition/NDCS/LG4000_test_filtered.txt']

        self.data_name = 'Notre Dame CrossSensor2013'
        self.test_type = 'Within'
        
    def gallery_loaderGet(self):
        return  self._root_path_gallery, self._gallery_list
    def probe_loaderGet(self):
        return  self._root_path_probe, self._probe_list