import pygame
from pygame.locals import *
import sys
import numpy as np
import pickle
sys.path.append('/Users/haomingyang/Documents/GitHub/INDOOR/indoor_final/prepare_data')
from prepare_for_rendering import take_away_dict_postprocess
#from myEnv.util.config_3d import Config
import scipy.spatial as spt
import paramiko
import os
import cv2 as cv

WIDTH = 1035
HEIGHT = 900

FPS = 30

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
SCALE = 0.45
HORIZONTAL_SCALE = 1
VERTICAL_SCALE = 1


RED = (255, 0, 0, 120)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)

NUM_UAV = 4
UAV_COLOR = [
    (0, 0, 0),       # Black
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Green (Dark)
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 128), # Gray
    (192, 192, 192), # Silver
    (255, 165, 0),   # Orange
    (210, 180, 40),  # Gold
    (138, 43, 226),  # Blue-Violet
    (255, 69, 0)     # Orange-Red
]


TASK_LIST = [[0, 0, 10, 10], [0, 10, 10, 20], [0, 20, 10, 30], [0, 30, 10, 40], [0, 40, 10, 50], [0, 50, 10, 60], [10, 0, 20, 10], [10, 10, 20, 20], [10, 20, 20, 30], [10, 30, 20, 40], [10, 40, 20, 50], [10, 50, 20, 60], [20, 0, 30, 10], [20, 10, 30, 20], [20, 20, 30, 30], [20, 30, 30, 40], [20, 40, 30, 50], [20, 50, 30, 60], [30, 0, 40, 10], [30, 10, 40, 20], [30, 20, 40, 30], [30, 30, 40, 40], [30, 40, 40, 50], [30, 50, 40, 60], [40, 0, 50, 10], [40, 10, 50, 20], [40, 20, 50, 30], [40, 30, 50, 40], [40, 40, 50, 50], [40, 50, 50, 60], [50, 0, 60, 10], [50, 10, 60, 20], [50, 20, 60, 30], [50, 30, 60, 40], [50, 40, 60, 50], [50, 50, 60, 60]]

TASK_ORDER = [2, 5, 0, 6]
ALLOCATED = False
RENDER_INCREASE = False
RENDER_AOI  = False
OBSTACLE= [[28.39,6.64,36.87,8.85],
                    [15.15,16.56,18.44,22.12],
                    [9.96,28.32,13.64,30.90],
                    [28.58,21.98,36.32,27.25],
                    [44.25,26.14,50.44,28.72],
                    [65.13,28.40,74.01,32.89],
                    [20.02,34.88,34.99,38.53]]
INDEX_CONVERT = []

FLOOR_NUM = 3 #换新图的时候 记得改这个


class UAV(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, trajectory, screen, index=0):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale(pygame.image.load("./intermediate/pictures/uav.png"), [10, 10])
        self.rect = self.image.get_rect()
        self.screen = screen
        self.floor = trajectory[0][0]
        self.rect.center = (trajectory[0][1][0] * SCALE, trajectory[0][1][1] * SCALE)
        self.trajectory = trajectory
        self.step = 0
        self.font = pygame.font.Font(None, int(20 * SCALE)) #写uav文字的 default，
        self.index = index

    def update(self):
        self.rect.x = self.trajectory[self.step][1][0] * SCALE
        self.rect.y = self.trajectory[self.step][1][1] * SCALE
        # print(self.rect)

        self.screen.blit(
            self.font.render('uav:{}'.format(self.index + 1), True, (0, 0, 0), (255, 255, 255)),
            (self.rect.x, self.rect.y))

        for j in range(self.step - 1):
            a = (self.trajectory[j][1][0] * SCALE, self.trajectory[j][1][1] * SCALE)
            b = (self.trajectory[j + 1][1][0] * SCALE, self.trajectory[j + 1][1][1] * SCALE)
            d = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
            if d(a,b) > 50 or d(a,b) == 0:
                # 控制划线变量在图中画出
                continue
            pygame.draw.line(self.screen, UAV_COLOR[self.index],a,b, 4)

        self.step += 1
        if self.step == len(self.trajectory):
            self.step = 0


class PoIs(pygame.sprite.Sprite):
    # sprite for the Player
    # UAV的位置就是所有图中的位置
    def __init__(self, info, screen,full_info):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale( pygame.image.load("./intermediate/pictures/poi_render.png"), [1, 1])
        self.rect = self.image.get_rect()
        self.screen = screen
        self.info = info
        self.step = 0
        self.radius = 0.125 * SCALE
        self.size = (0.5 * SCALE, 0.5 * SCALE)
        self.font = pygame.font.Font(None, int(10 * SCALE))
        self.full_info = full_info
        if RENDER_INCREASE:
            self.poi_speed = full_info['poi_arrival']
        
        

    def update(self):

        # for index in range(len(OBSTACLE)):
        #     x1,y1,x2,y2=OBSTACLE[index][0]*SCALE,OBSTACLE[index][1]*SCALE,OBSTACLE[index][2]*SCALE,OBSTACLE[index][3]*SCALE
        #     pygame.draw.polygon(self.screen, (255,0,0,255),
        #                                 [ [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        #                             )


        for fl in self.info[self.step]['val'].keys(): # 符合floor_list的顺序
            for index in range(len(self.info[self.step]['val'][fl])):
                if self.info[self.step]['val'][fl][index] > 0:
                    color = (0, 0, 255, min(255, (self.info[self.step]['val'][fl][index]) * 10.5))
                    # pygame.draw.circle(self.screen, color,
                    #                 (self.info[self.step]['pos'][fl][index][0],
                    #                     self.info[self.step]['pos'][fl][index][1]),
                    #                 self.radius)
                    
                    if RENDER_INCREASE:
                        color = (255, 0,0, min(255, (self.poi_speed[index,self.step]) * 20.5))
                        pygame.draw.circle(self.screen, color,
                                    (self.info[self.step]['pos'][fl][index][0] * SCALE,
                                        self.info[self.step]['pos'][fl][index][1]* SCALE),
                                    self.radius+3)
                    
                    else:
                        if RENDER_AOI:
                            self.screen.blit(
                            self.font.render(str(round(self.info[self.step]['aoi'][fl][index]/20, 1)), True, (0, 0, 0),
                                            (255, 255, 255)),
                            (self.info[self.step]['pos'][fl][index][0] * SCALE, self.info[self.step]['pos'][fl][index][1] * SCALE))
                        else:
                            #print(self.info[self.step]['pos'][fl][index])
                            self.screen.blit(self.font.render(str(round(self.info[self.step]['val'][fl][index], 1)), True, (0, 0, 0),
                                            (255, 255, 255)),
                            (self.info[self.step]['pos'][fl][index][0] * SCALE, self.info[self.step]['pos'][fl][index][1] * SCALE)) # 由于cv2和pygame的坐标方式不太一样，反过来了

                

        self.step += 1
        if self.step == len(self.info):
            self.step = 0


class Task(pygame.sprite.Sprite):
    # sprite for the Player
    def __init__(self, info, screen, task_info=None):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.transform.smoothscale(pygame.image.load("./intermediate/pictures/concatenate_for_render.png'"), [1, 1])
        self.rect = self.image.get_rect()
        self.screen = screen
        self.info = info
        self.step = 0
        self.size = (0.5 * SCALE, 0.5 * SCALE)
        self.radius = 0.5 * SCALE
        self.font = pygame.font.Font(None, 100*SCALE)
        self.remark = False
        self.task_info = task_info
        self.task_radius = 5 * SCALE

    def update(self):

        # selected = self.task_info[self.step]
        # 
        # task_radius = 5 * SCALE
        # center_x, center_y = (AREA_LIST[0][selected][0] + AREA_LIST[0][selected][2]) / 2 * SCALE, (
        #         AREA_LIST[0][selected][1] + AREA_LIST[0][selected][3]) / 2 * SCALE
        # 
        # x1 = center_x - task_radius
        # x2 = center_x + task_radius
        # y1 = center_y - task_radius
        # y2 = center_y + task_radius
        # #print(center_x, center_y)
        # color = (0, 0, 255, 125)
        # pygame.draw.polygon(self.screen, color, [
        #     [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
        # ])
        # 
        if ALLOCATED:
            for index, selected in enumerate(self.task_info[self.step]):
                center_x, center_y = (TASK_LIST[selected][0] + TASK_LIST[selected][2]) / 2 * SCALE, (
                        TASK_LIST[selected][1] + TASK_LIST[selected][3]) / 2 * SCALE
                # if selected <= 8:
                #     center_x, center_y = (TASK_LIST[selected][0] + TASK_LIST[selected][2]) / 2 * SCALE, (
                #                 TASK_LIST[selected][1] + TASK_LIST[selected][3]) / 2 * SCALE
                # else:
                #     if self.step >= TASK_START:
                #         s = TASK_ORDER[(self.step - TASK_START) // 50]
                #         center_x, center_y = (TASK_LIST[s][0] + TASK_LIST[s][2]) / 2 * SCALE, (
                #                     TASK_LIST[s][1] + TASK_LIST[s][3]) / 2 * SCALE
                #     else:
                #         center_x, center_y = -10000, -10000
                x1 = center_x - self.task_radius
                x2 = center_x + self.task_radius
                y1 = center_y - self.task_radius
                y2 = center_y + self.task_radius
                # if selected <= 7:
                #    color = (UAV_COLOR[index][0], UAV_COLOR[index][1], UAV_COLOR[index][2], 100)
                pygame.draw.polygon(self.screen, (255,0,0,100),
                                       [ [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
                                        )
                # else:
                #     color = (255,0,0,100)

        # for index in range(self.info[self.step]['location'].shape[0]):
        #     # color = ((10 - self.info[self.step]['val'][index]) * 25.5, (10 - self.info[self.step]['val'][index]) * 25.5,
        #     #        (10 - self.info[self.step]['val'][index]) * 25.5)
        #
        #     value = self.info[self.step]['value'][index]
        #     x1 = self.info[self.step]['location'][index][0] * SCALE - self.radius
        #     x2 = self.info[self.step]['location'][index][0] * SCALE + self.radius
        #     y1 = self.info[self.step]['location'][index][1] * SCALE - self.radius
        #     y2 = self.info[self.step]['location'][index][1] * SCALE + self.radius
        #
        #     if value > 0:
        #         pygame.draw.polygon(self.screen, (255, 0, 0, (value / 10) * 255), [
        #             [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
        #         ])
        #         if self.remark:
        #             self.screen.blit(
        #                 self.font.render('value:{:.0f}'.format(value), True, (0, 0, 0), (255, 255, 255)),
        #                 (self.info[self.step]['location'][index][0] * SCALE,
        #                  self.info[self.step]['location'][index][1] * SCALE))
        #
        #             self.screen.blit(
        #                 self.font.render('index:{}'.format(index), True, (0, 0, 0), (255, 255, 255)),
        #                 (self.info[self.step]['location'][index][0] * SCALE,
        #                  self.info[self.step]['location'][index][1] * SCALE + 15 * SCALE / 20))

        self.step += 1
        if self.step == len(self.info):
            self.step = 0


class RenderEnv(object):

    def __init__(self, info):
        pygame.init()
        pygame.display.set_caption('Indoor MCS')
        self.info = info
        self.clock = pygame.time.Clock()
        self.bg = pygame.image.load('./intermediate/pictures/concatenate_for_render.png')
        self.bg = pygame.transform.scale(self.bg, (self.bg.get_width() * SCALE, self.bg.get_height() * SCALE))
        print(self.bg.get_height(), self.bg.get_width()) # width = 900 * 2 height = 1035
        self.windows = pygame.display.set_mode(((WIDTH-150) * FLOOR_NUM * SCALE, (HEIGHT+30) * SCALE))
        self.screen = pygame.Surface((WIDTH * FLOOR_NUM * SCALE, HEIGHT*SCALE), pygame.SRCALPHA)
        self.all = pygame.sprite.LayeredUpdates()
        self.font = pygame.font.Font(None, int(50 * SCALE)) # 文字大小
        self.small_font = pygame.font.Font(None, int(100 * SCALE))
        self.step = 0
        self.total_reward = 0
        self.render_allocated = ALLOCATED
        self.task_allocated = None
        # if self.render_allocated:
        # self.task_allocated = info['selected_area']

        self.NUM_STEP = len(info['reward_history'][0])
        self.NUM_UAV = len(self.info['uav_trace'])

        self.init_sprite()

    def init_sprite(self):
        self.PoIs = PoIs(self.info['poi_history'], self.screen,self.info)
        self.all.add(self.PoIs)
        for i in range(self.NUM_UAV):
            uav = UAV(self.info['uav_trace'][i], self.screen, i)
            self.all.add(uav)

    def start(self):
        Rendering = True
        paused = False

        while Rendering:
            self.windows.fill(WHITE)
            self.screen.fill(WHITE)
            self.clock.tick(FPS)
            self.screen.blit(self.bg, (0,0))
            # Process input (events)

            for event in pygame.event.get():
                # check for closing window
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        pygame.quit()
                        return
                    if event.key == K_SPACE:
                        paused = not paused

            if not paused:
                # Update
                self.all.update()

                # Draw / render

                self.all.draw(self.screen)

                # *after* drawing everything, flip the display
                self.windows.blit(self.screen, (0,0))
                self.render_info()
                self.step += 1
                if self.step == self.NUM_STEP:
                    self.total_reward = 0
                    self.step = 0
                pygame.display.update()

    def render_info(self):
        txt = []
        txt.append(self.font.render('step: ' + str(self.step), True, (0, 0, 0), (255, 255, 255)))
        txt.append(self.font.render('collection_ratio: ' + str(np.round(self.info['collection_ratio'][0][self.step], 3)), True, (0, 0, 0), (255, 255, 255)))
        txt.append(self.font.render('violation_ratio: ' + str(np.round(self.info['violation_ratio'][self.step], 3)), True, (0, 0, 0), (255, 255, 255)))
        # txt.append(
        #     self.font.render('total_reward:{:.2f}'.format(self.total_reward), True, (0, 0, 0),
        #                      (255, 255, 255)))
        # reward = self.info['reward_history'][self.step]
        # self.total_reward += reward

  
        # if self.NUM_UAV > 10:

        #     uav_reward = ' '.join([str(round(r, 1)) for r in self.info['uav_reward'][self.step]])

        #     task_reward = ' '.join([str(round(r, 1)) for r in self.info['task_reward'][self.step]])
        #     txt.append(
        #         self.small_font.render('uav_reward:{}'.format(uav_reward), True, (0, 0, 0),
        #                                (255, 255, 255)))
        #     txt.append(
        #         self.small_font.render('task_reward:{}'.format(task_reward), True, (0, 0, 0),
        #                                (255, 255, 255)))
        # else:
        #     txt.append(
        #         self.font.render('uav_reward:{}'.format(round(reward, 2)), True, (0, 0, 0),
        #                          (255, 255, 255)))
        #     txt.append(
        #         self.font.render('total:{}'.format(round(self.total_reward, 2)), True, (0, 0, 0),
        #                          (255, 255, 255)))

            # selected = self.info['selected_area'][self.step]

            # txt.append(
            #     self.font.render('area:{}'.format(round(selected, 2)), True, (0, 0, 0),
            #                      (255, 255, 255)))

        # if self.render_allocated:
        #     txt.append(
        #         self.small_font.render(
        #             'task_allocated:' + " ".join([str(i) for i in self.task_allocated[self.step]]), True,
        #             (0, 0, 0),
        #             (255, 255, 255)))
        for index, t in enumerate(txt):
            self.windows.blit(t, (0, index * 40 * SCALE))


def get_convex_hull():
    config = Config()
    poi_position = np.array(config("poi_position"))
    task_position = np.array(config("task_position"))
    allocated = []
    for group_index in range(poi_position.shape[0]):
        g = []
        hull = spt.ConvexHull(points=poi_position[group_index, :, :])
        for vertice in hull.vertices:
            g.append(vertice)
        allocated.append(poi_position[group_index, g, :] * SCALE * config("map_x"))
    return allocated


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


if __name__ == '__main__':
    # path = '/home/liuchi/zqr/indoor_final/intermediate/data/generate_dict.pickle'
    # client = paramiko.SSHClient()
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # client.connect("10.1.114.77", 22, "liuchi", "LIUCHI-linc-2021!", timeout=5)
    # sftp_client = client.open_sftp()
    # remote_file = sftp_client.open(path, 'r')
    # in_dict = pickle.load(remote_file)

    # path = '/home/liuchi/wh/adeptRL/adept/env/env_bound/default_1.txt'
    # #INDEX_CONVERT = get_convex_hull()
    # client = paramiko.SSHClient()
    # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # client.connect("10.1.114.77", 22, "liuchi", "LIUCHI-linc-2021!", timeout=5)
    # sftp_client = client.open_sftp()
    # remote_file = sftp_client.open(path, 'r')
    # info = pickle.load(remote_file)
    # pickle.dump(info, open('./whinfo.pickle','wb'))
    # remote_file.close()
    # print(info['y_poi_0_visit_tims'])
    # path = '/Users/wanghao/wh/Code/Python/mcs_dynamically/manager_1653191380.txt'
    # info = load_variavle(path)
    # if 'manager' in path:
    #     ALLOCATED = True
    
    # info = pickle.load(open('./whinfo.pickle','rb'))
    
    with open('./intermediate/data/generate_dict.pickle', 'rb') as f:
        in_dict = pickle.load(f)
        
    info = take_away_dict_postprocess(in_dict)
    R = RenderEnv(info)
    R.start()



