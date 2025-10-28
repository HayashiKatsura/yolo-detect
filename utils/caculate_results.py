from utils.ReadTxt import read_txt


def calculate_results_txt(result_txt):
    macro_result =read_txt(result_txt)
    no_detections = 0
    total_detections = len(macro_result)
    # total_detections = 650
    for item in macro_result:
        try:
            item_info = item.replace(' ', '')
            item_info = item_info.split('640x640')[1]
            item_info = item_info.split(',')[0]
            if 'no' in item_info:
                no_detections += 1
        except:
            continue
    print(f"Total detections: {total_detections}")
    print(f"No detections: {no_detections}")
    print(f"No Detection rate: {(1-(total_detections - no_detections) / total_detections) * 100:.2f}%")



if __name__ == '__main__':
    result_txt = r'D:\Coding\ultralytics\A_project\CustomTrain\PREDICT\1MCustomTrain_01081742_VAL_TRAIN\1.txt'
    calculate_results_txt(result_txt)