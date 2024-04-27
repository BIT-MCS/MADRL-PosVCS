import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image

sys.path.append('/Users/haomingyang/Documents/GitHub/INDOOR/indoor_final')
from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, visualize_position, visualize_floor_plan, save_figure_to_html, save_figure_to_png
from obstacle import *
from obstacle import _visualize_poi_on_map, _gridline_matching
from data_process import *
from MAPPO.env_config.default_config import config

FL = 'F2'
site = 'site3'

floor_data_dir = f'./data/{site}/' + FL # Change this to change floor
path_data_dir = floor_data_dir + '/path_data_files'


floor_plan_filename = floor_data_dir + '/floor_image.png'
floor_info_filename = floor_data_dir + '/floor_info.json'
floor_plan_modified_save_dir = './output/mod_floorplan'

save_dir = f'./output/{site}/'+ FL
poi_img_save_dir = save_dir + '/poi_images'
path_image_save_dir = save_dir + '/path_images'
step_position_image_save_dir = save_dir
magn_image_save_dir = save_dir
wifi_image_save_dir = save_dir + '/wifi_images'
ibeacon_image_save_dir = save_dir + '/ibeacon_images'
wifi_count_image_save_dir = save_dir


def calibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
    mwi_datas = {}
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')

        path_datas = read_data_file(path_filename)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon
        posi_datas = path_datas.waypoint

        step_positions = compute_step_positions(acce_datas, ahrs_datas, posi_datas)
        # visualize_trajectory(posi_datas[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Position', show=True)

        if wifi_datas.size != 0:
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['wifi'] = np.append(mwi_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': wifi_ds,
                        'ibeacon': np.zeros((0, 3))
                    }

        if ibeacon_datas.size != 0:
            sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
            ibeacon_datas_list = split_ts_seq(ibeacon_datas, sep_tss)
            for ibeacon_ds in ibeacon_datas_list:
                diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['ibeacon'] = np.append(mwi_datas[target_xy_key]['ibeacon'], ibeacon_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': np.zeros((0, 5)),
                        'ibeacon': ibeacon_ds
                    }

        sep_tss = np.unique(magn_datas[:, 0].astype(float))
        magn_datas_list = split_ts_seq(magn_datas, sep_tss)
        for magn_ds in magn_datas_list:
            diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]['magnetic'] = np.append(mwi_datas[target_xy_key]['magnetic'], magn_ds, axis=0)
            else:
                mwi_datas[target_xy_key] = {
                    'magnetic': magn_ds,
                    'wifi': np.zeros((0, 5)),
                    'ibeacon': np.zeros((0, 3))
                }

    return mwi_datas


def extract_magnetic_strength(mwi_datas):
    magnetic_strength = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        magnetic_data = mwi_datas[position_key]['magnetic']
        magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_strength[position_key] = magnetic_s

    return magnetic_strength


def extract_wifi_rssi(mwi_datas):
    wifi_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])
            if bssid in wifi_rssi:
                position_rssi = wifi_rssi[bssid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            wifi_rssi[bssid] = position_rssi

    return wifi_rssi

def wifi_time_rssi(mwi_datas):
    # count for a given position and a wifi_id, any changes made upon time
    wifi_rssi = {} #dict type: {(pos, bssid):[(time_stamp,rssi)]}
    var_wifi = []
    for position_key in mwi_datas:
        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])
            time_stamp = wifi_d[0]
            if (position_key, bssid) not in wifi_rssi:
                 wifi_rssi[(position_key, bssid)] = [(time_stamp,rssi)]
            else:
                 wifi_rssi[(position_key, bssid)].append((time_stamp,rssi))
                 if bssid not in var_wifi:
                    var_wifi.append(bssid)
                 #print(wifi_rssi[(position_key, bssid)])
    # print(len(var_wifi))
    # print(len(wifi_rssi))
    return wifi_rssi

def bssid_pos_rssi(mwi_datas):
    # count for a given position and a wifi_id, any changes made upon time
    wifi_rssi = {} #dict type: {bssid: {pos: [rssi, rssi] }}
    var_wifi = []
    for position_key in mwi_datas:
        wifi_data = mwi_datas[position_key]['wifi']
        for wifi_d in wifi_data:
            bssid = wifi_d[2]
            rssi = int(wifi_d[3])
            time_stamp = wifi_d[0]
            if bssid not in wifi_rssi:
                 wifi_rssi[bssid] = {}
            if position_key not in wifi_rssi[bssid]:
                wifi_rssi[bssid][position_key] = [rssi]
            else:
                 wifi_rssi[bssid][position_key].append(rssi)
    #print(wifi_rssi)
    return wifi_rssi

def wifi_bssi_process(process_rssi):
    # we should split a radio map by time steps
    # a dict {(bssid, timestep) : [(pos1,rssi1), (pos2,rssi2),...]}
    wifi_rssi_dict = {}
    for ks in list(process_rssi.keys()):
        pos = ks[0]
        bssid = ks[1]
        for vals in process_rssi[ks]:
            timestep = vals[0]
            rssi = vals[1]
            if (bssid,timestep) not in wifi_rssi_dict:
                wifi_rssi_dict[(bssid,timestep)] = [(pos, rssi)]
            else:
                wifi_rssi_dict[(bssid,timestep)].append((pos, rssi))
    #print(len(wifi_rssi_dict.keys()))
    return wifi_rssi_dict

def extract_ibeacon_rssi(mwi_datas):
    ibeacon_rssi = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')
        ibeacon_data = mwi_datas[position_key]['ibeacon']
        for ibeacon_d in ibeacon_data:
            ummid = ibeacon_d[1]
            rssi = int(ibeacon_d[2])

            if ummid in ibeacon_rssi:
                position_rssi = ibeacon_rssi[ummid]
                if position_key in position_rssi:
                    old_rssi = position_rssi[position_key][0]
                    old_count = position_rssi[position_key][1]
                    position_rssi[position_key][0] = (old_rssi * old_count + rssi) / (old_count + 1)
                    position_rssi[position_key][1] = old_count + 1
                else:
                    position_rssi[position_key] = np.array([rssi, 1])
            else:
                position_rssi = {}
                position_rssi[position_key] = np.array([rssi, 1])

            ibeacon_rssi[ummid] = position_rssi

    return ibeacon_rssi


def extract_wifi_count(mwi_datas):
    wifi_counts = {}
    for position_key in mwi_datas:
        # print(f'Position: {position_key}')

        wifi_data = mwi_datas[position_key]['wifi']
        count = np.unique(wifi_data[:, 2]).shape[0]
        wifi_counts[position_key] = count

    return wifi_counts

def POI_generate():
    # dics 的结构：{bssid, {pos: [rssi,rssi]}}
    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)
    dics = bssid_pos_rssi(mwi_datas)
    ret = []
    for d in dics.values():
        ret.extend(list(d.keys()))
    return list(set(ret))
            
def POI_matchup(fig, x_max, y_max, poi_list, map_list):
    # plotly 的feature的轴长查找用：fig['layout']['xaxis']['range']， fig['layout']['yaxis']['range']
    # 例子：返回（0，250）
    x_range = fig['layout']['xaxis']['range']
    y_range = fig['layout']['yaxis']['range']
    x_low = float(x_range[0])
    x_up = float(x_range[1])
    y_low = float(y_range[0])
    y_up = float(y_range[1])
    #print(x_up,y_up,fig,'hiiiii')
    d = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2

    # 将POI坐标对应到像素点上去
    ret = []
    for points in poi_list:
        xnew = (float(points[0]) / x_up) * x_max
        ynew = (float(points[1])/ y_up) * y_max
        min_d = d((xnew,ynew), map_list[0])
        modified = map_list[0]
        for mapp in map_list:
            temp = d((xnew,ynew), mapp)
            #print(temp, (xnew, ynew))
            if temp < min_d:
                min_d = temp
                modified = mapp
        ret.append(modified)
        #print(xnew,ynew, points, modified)
    return list(set(ret)) 

def map_points(xmax, ymax, step = config['poi_distance']):
    # 返回所有的地图标点
    #whites = obstacle._find_all_white(dir)
    ret = []
    for i in range(int(xmax / step)):
        for j in range(int(ymax/step)):
            ret.append((i * step,j * step))
    #print(ret)
    return ret


if __name__ == "__main__":

    assert 'prepare_data' in os.getcwd() # 请在prepare_data 目录下运行

    Path(path_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(magn_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(wifi_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(ibeacon_image_save_dir).mkdir(parents=True, exist_ok=True)

    with open(floor_info_filename) as f:
        floor_info = json.load(f)
    width_meter = floor_info["map_info"]["width"] # 从floor info中抠出来了
    height_meter = floor_info["map_info"]["height"] # 从floor info中抠出来了

    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))
 
    fig = visualize_floor_plan(floor_plan_filename, width_meter, height_meter, title=f'{FL} Floor Plan', show=True)
    png_filename = floor_plan_modified_save_dir + f'/{site}/{FL}.png'
    save_figure_to_png(fig, png_filename)


    # # 1. visualize ground truth positions
    # print('Visualizing ground truth positions...')
    # for path_filename in path_filenames:
    #     print(f'Processing file: {path_filename}...')

    #     path_data = read_data_file(path_filename)
    #     path_id = path_filename.name.split(".")[0]
    #     fig = visualize_trajectory(path_data.waypoint[:, 1:3], floor_plan_filename, width_meter, height_meter, title=path_id, show=False)
    #     html_filename = f'{path_image_save_dir}/{path_id}.html'
    #     html_filename = str(Path(html_filename).resolve())
    #     save_figure_to_html(fig, html_filename)

    # # 2. visualize step position, magnetic, wifi, ibeacon
    # print('Visualizing more information...')
    #path_filenames = ['./combine.txt']
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)
    #dics = wifi_time_rssi(mwi_datas)
    dics = bssid_pos_rssi(mwi_datas)
    #wifi_bssi_process(dics) 

    # step_positions = np.array(list(mwi_datas.keys()))
    # fig = visualize_trajectory(step_positions, floor_plan_filename, width_meter, height_meter, mode='markers', title='Step Position', show=True)
    # html_filename = f'{step_position_image_save_dir}/step_position.html'
    # html_filename = str(Path(html_filename).resolve())
    # save_figure_to_html(fig, html_filename)

    # magnetic_strength = extract_magnetic_strength(mwi_datas)
    # heat_positions = np.array(list(magnetic_strength.keys()))
    # heat_values = np.array(list(magnetic_strength.values()))
    # fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='mu tesla', title='Magnetic Strength', show=True)
    # html_filename = f'{magn_image_save_dir}/magnetic_strength.html'
    # html_filename = str(Path(html_filename).resolve())
    # save_figure_to_html(fig, html_filename)

    wifi_rssi = extract_wifi_rssi(mwi_datas)
    print(f'This floor has {len(wifi_rssi.keys())} wifi aps')

    one_wifi_bssid = list(wifi_rssi.keys())[0:1]
    ten_wifi_bssids = list(wifi_rssi.keys())[0:10]
    all_wifi_bssids = list(wifi_rssi.keys())

    position_data = {} #用来统计一个地方能收集到多少个数据，无视时间段
    position_AP_counts = {} #用来统计一个地方能收集到多少个不同的AP

    for AP in all_wifi_bssids:
        for pos in wifi_rssi[AP].keys():
            if pos not in position_data.keys():     # multiple-rssi exist
                position_data[pos] = 1
                #print(wifi_rssi[AP][pos])
                position_AP_counts[pos] = set([wifi_rssi[AP][pos][0]])
            else:
                position_data[pos] += 1
                position_AP_counts[pos].add(wifi_rssi[AP][pos][0])
        
    
    heat_positions = np.array(list(position_data.keys()))
    heat_values = np.array(list(position_data.values()))
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='#', title=f'{FL} # data', show=True)
    html_filename = str(FL + 'num_count.png')
    save_figure_to_png(fig, html_filename)

    heat_positions = np.array(list(position_AP_counts.keys()))
    from site2_handcraft import *
    #heat_positions = site2_F4(heat_positions)
    # img = cv.imread(f'./output/mod_floorplan/{site}/{FL}_vis.png')
    # print(img.shape, 'jjjjj', width_meter, height_meter)
    # for pos in heat_positions:
    #     # 原点：（870,80）左上顶点：（180，80）右下顶点 （870，820）右上顶点（180,820）
    #     #img[int((height_meter - pos[1]) * 742/height_meter)][int(pos[0] * 800/width_meter)] = np.array([0,0,0])
    #     # img[870-c][820] = np.array([255,0,0])

    #     # 转化规则如下：
    #     x = int((height_meter - pos[1]) * 742/height_meter) 
    #     y = int(pos[0] * 800/width_meter)
    #     x = int(180 + 740/800 * x)
    #     y = int(80 + 690/742 * y)

    #     img[x][y] = np.array([0,0,255])
        
    # cv.imwrite(('./combinealltest.png'), img)
    POIs = _gridline_matching(f'./output/mod_floorplan/{site}/{FL}_bb.png', heat_positions, width_meter, height_meter)
    
    heat_values = np.array(len(list(position_AP_counts.values())))
    # print(width_meter,height_meter,'lllll')
    # width_meter = width_meter * 1035/742
    # height_meter = height_meter * 900/800

    # imgs = cv.imread(f'./data/{site}/{FL}/floor_image_prime.png')
    # print(width_meter,height_meter) # 236.71181213998395 219.74676479990106
    # c = 0
    # for pos in list(position_AP_counts.keys()):
    #     imgs[int((height_meter - pos[1]) * 742/height_meter)][int(pos[0] * 800/width_meter)] = np.array([255,0,0]) # 长轴也要翻转
    #     #imgs[100][10 + c] = np.array([255,0,0])
    #     c +=1
    # cv.imwrite('./checksss.png',imgs)

    fig = visualize_position(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, show=True)
    html_filename = str(FL + 'num_AP.png')
    save_figure_to_png(fig, html_filename)

    img = cv.imread(f'./output/mod_floorplan/{site}/{FL}_vis.png')
    
    poi_c = 0
    for pos in POIs.keys():
        if len(POIs[pos]) > 0:
            poi_c += 1
            img[pos[0]][pos[1]] = np.array([0,0,255])
    print('Total POI: ', poi_c)
    cv.imwrite((f'./output/mod_floorplan/{site}/{FL}_POI_on_grid.png'), img)



    def generate_time_difference_waypoints(dic) :
        #地图上所有多次测量差距大于2dbm的点的集合可以作为POI，然后我们还可以考虑AOI，比如我们相信不同时段的信号
        all_waypoints = {}
        for bssid in list(dic.keys()):
            for pos in list(dic[bssid].keys()):
                if len(dic[bssid][pos]) > 2 : 
                    all_waypoints[pos] = max(dic[bssid][pos]) - min(dic[bssid][pos])
        print(all_waypoints)
        
        heat_positions = np.array(list(all_waypoints.keys()))
        heat_values = np.array(list(all_waypoints.values()))
        fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='delta_dBm', title=f'{FL} TD RSSI', show=True)
        #print(fig['layout']['xaxis']['range'])
        html_filename = str(FL + ' waypoinypoi.html')
        save_figure_to_html(fig, html_filename)

        return list(all_waypoints.keys())
    
    # 生成所有POI的点
    pois = POI_matchup(fig, 1035,900, generate_time_difference_waypoints(dics), map_points(1035,900,config['poi_distance']))
    # 将POI点成绿色, 覆盖在FL_env.png 中，命名为 FL_POI.png
    
    op = _visualize_poi_on_map(floor_plan_modified_save_dir + '/'+ site +'/' + FL + '_env.png', pois)
    ig = Image.fromarray(op)
    ig.save(f'./output/mod_floorplan/{site}/{FL}_POI.png')
    org = cv.imread(f'./data/{site}/{FL}/floor_image.png')
    print(org.shape)
    '''
    print('Example 10 wifi ap bssids:\n')
    for bssid in ten_wifi_bssids:
        print(bssid)
    target_wifi = input(f"Please input target wifi ap bssid:\n")
    # target_wifi = '1e:74:9c:a7:b2:e4'
    heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
    heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'Wifi: {target_wifi} RSSI', show=True)
    html_filename = f'{wifi_image_save_dir}/{target_wifi.replace(":", "-")}.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)
    '''

    l_p = {} #{keys: position; values: [(rssi_1, wifi_id), (rssi_2, wifi_id),...]}

    for target_wifi in all_wifi_bssids:
        tmp = wifi_rssi[target_wifi]
        for i in range(len(list(tmp.keys()))):
            k = list(tmp.keys())[i]
            if k not in l_p.keys():      
                l_p[k] = [(tmp[k], target_wifi)]
            else:
                l_p[k].append((tmp[k], target_wifi))
    
    
    for pos in list(l_p.keys()):
        wifi_d = {}
        for itm in l_p[pos]:
            if itm[1] not in list(wifi_d.keys()):
                wifi_d[itm[1]] = itm[0]
            else:
                wifi_d[itm[1]].append(itm[0])
                print(wifi_d[itm[1]])
        

    heat_positions = np.array(list(l_p.keys()))
    heat_values = np.array([np.average([l_p[i][j][0] for j in range(len(l_p[i]))]) for i in list(l_p.keys())])
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='avg_dBm', title=f'{FL} Avergae RSSI', show=True)
    html_filename = str(FL + ' waypoints.html')
    png_filename = str(FL + ' waypoints.png')
    save_figure_to_html(fig, html_filename)
    save_figure_to_png(fig, png_filename)
        
    
    # for target_wifi in all_wifi_bssids:
    #     heat_positions = np.array(list(wifi_rssi[target_wifi].keys()))
    #     heat_values = np.array(list(wifi_rssi[target_wifi].values()))[:, 0]
    #     fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'Wifi: {target_wifi} RSSI', show=True)
    #     html_filename = f'{wifi_image_save_dir}/{target_wifi.replace(":", "-")}.html'
    #     html_filename = str(Path(html_filename).resolve())
    #     save_figure_to_html(fig, html_filename)


    # ibeacon_rssi = extract_ibeacon_rssi(mwi_datas)
    # print(f'This floor has {len(ibeacon_rssi.keys())} ibeacons')
    # ten_ibeacon_ummids = list(ibeacon_rssi.keys())[0:10]
    # all_ibeacon_ummids = list(ibeacon_rssi.keys())[0:10]
    
    # for target_ibeacon in all_ibeacon_ummids:
    #     heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
    #     heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]
    #     fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'iBeacon: {target_ibeacon} RSSI', show=True)
    #     html_filename = f'{ibeacon_image_save_dir}/{target_ibeacon}.html'
    #     html_filename = str(Path(html_filename).resolve())
    #     save_figure_to_html(fig, html_filename)

    '''
    print('Example 10 ibeacon UUID_MajorID_MinorIDs:\n')
    for ummid in ten_ibeacon_ummids:
        print(ummid)
    target_ibeacon = input(f"Please input target ibeacon UUID_MajorID_MinorID:\n")
    # target_ibeacon = 'FDA50693-A4E2-4FB1-AFCF-C6EB07647825_10073_61418'
    heat_positions = np.array(list(ibeacon_rssi[target_ibeacon].keys()))
    heat_values = np.array(list(ibeacon_rssi[target_ibeacon].values()))[:, 0]
    fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='dBm', title=f'iBeacon: {target_ibeacon} RSSI', show=True)
    html_filename = f'{ibeacon_image_save_dir}/{target_ibeacon}.html'
    html_filename = str(Path(html_filename).resolve())
    save_figure_to_html(fig, html_filename)
    '''

    # wifi_counts = extract_wifi_count(mwi_datas)
    # heat_positions = np.array(list(wifi_counts.keys()))
    # heat_values = np.array(list(wifi_counts.values()))
    # # filter out positions that no wifi detected
    # mask = heat_values != 0
    # heat_positions = heat_positions[mask]
    # heat_values = heat_values[mask]
    # fig = visualize_heatmap(heat_positions, heat_values, floor_plan_filename, width_meter, height_meter, colorbar_title='number', title=f'Wifi Count', show=True)
    # html_filename = f'{wifi_count_image_save_dir}/wifi_count.html'
    # html_filename = str(Path(html_filename).resolve())
    # save_figure_to_html(fig, html_filename)

    print('fff')
