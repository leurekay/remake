import pandas as pd
import glob
 
import requests
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def download_image(url, save_dir):
    """
    下载图片并保存到指定目录。
    
    参数：
    - url: 图片的URL
    - save_dir: 保存图片的目录
    """
    try:
        # 获取图片名称
        image_name = url.split('/')[-1]
        save_path = os.path.join(save_dir, image_name)
        
        # 发送HTTP请求获取图片
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 打开文件以写入二进制数据
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        
        # print(f"Image successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images(url_list, save_dir, max_workers=10):
    """
    使用多线程下载图片。
    
    参数：
    - url_list: 图片的URL列表
    - save_dir: 保存图片的目录
    - max_workers: 最大线程数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_url = {executor.submit(download_image, url, save_dir): url for url in url_list}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {url}: {e}")


def read_csv_fields(file_path, target_fields=None):
    """
    读取CSV文件并提取指定的目标字段列表。
    
    参数：
    - file_path: CSV文件路径
    - target_fields: 目标字段列表
    
    返回：
    - 包含目标字段的数据框
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 如果目标字段为空或None，则返回所有字段
        if not target_fields:
            return df

        # 检查目标字段是否都存在于CSV文件中
        for field in target_fields:
            if field not in df.columns:
                raise ValueError(f"Field '{field}' not found in the CSV file.")
        
        # 提取目标字段
        result_df = df[target_fields]
        
        return result_df
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def print_full_dataframe(df,index=-1):
    """
    打印完整的DataFrame。
    
    参数：
    - df: 要打印的DataFrame
    """

    
    if index>=0:
        for field in df.columns:
            print("{} : {}".format(field,df.loc[index,field]))
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
            

if __name__=="__main__":
    path_list=glob.glob(r"data/csv/*.csv")
    output_dir=r"data/image"
    print(path_list)

    for path in path_list:
        file_name = os.path.basename(path)
        file_name, _ = os.path.splitext(file_name)
        
        sub_output_dir=os.path.join(output_dir,file_name)
        df=read_csv_fields(path)
        url_list=df.loc[:,"original_image"].to_list()
        random.shuffle(url_list)

        print("task {}, {} images need download".format(file_name,len(url_list)))
        t1=time.time()
        download_images(url_list[:9990],sub_output_dir)
        t2=time.time()
        N_images=len(os.listdir(sub_output_dir))
        print("{} : dowload {} images, consume {} s, average time {} s ".format(file_name,N_images,t2-t1,(t2-t1)/N_images))
        time.sleep(60)