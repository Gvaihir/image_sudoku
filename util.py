# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class MetaData(object):
    """
    Class with image name, centroid coordinates, surface area

    """
    def __init__(self, im_name, points_ls, area_ls):
        self.image = im_name
        self.points = points_ls
        self.area = area_ls