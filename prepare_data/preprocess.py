# 这个文件用来整理所有预处理部分

from obstacle import *
from data_process import *
from main import *
from MAPPO.env_config.default_config import *

if __name__ == "__main__":
    # 环境只需要最终的 F3_POI_on_grids.png ...这种格式的

    # 需要先从main中生成F3.png 文件 目前大小应该设置为900* 1035
    obstacle._pipeline_processing('site2','F3')


