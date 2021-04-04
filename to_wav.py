import os

source_path = '/Users/shimingkun/Desktop/jupyter/Voice/deal'
save_path = '/Users/shimingkun/Desktop/jupyter/Voice/music_data'
source_file = os.listdir(source_path)

for source in source_file:
    new_name = source[:-4]  # 截取歌曲名字,删除后面的.m4a
    # print("ffmpeg -i " + m4a_path + m4a
    #           + " " + save_path + new_name + ".wav")
    print('正在转换歌曲:{}'.format(new_name))

    os.system("ffmpeg -i " + source_path + source + " " + save_path + new_name + ".wav")
print('所有歌曲转换完毕!')