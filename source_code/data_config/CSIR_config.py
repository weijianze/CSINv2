# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        ####################### testing dataset #########################
        # CSIR
        self._root_path_gallery = ['../../Data01_recognition/CSIR/']
        self._gallery_list = ['../../Data01_recognition/CSIR/test_gallery.txt']

        self._root_path_probe = ['../../Data01_recognition/CSIR/']
        self._probe_list = ['../../Data01_recognition/CSIR/test_probe.txt']

        self.data_name = 'CASIA Cross-sensor'
        self.test_type = 'Within'
        

    def gallery_loaderGet(self):
        return  self._root_path_gallery, self._gallery_list
    def probe_loaderGet(self):
        return  self._root_path_probe, self._probe_list