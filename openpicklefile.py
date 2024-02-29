# import os
# import os.path as osp

# dataset_root = '/media/ani/TOSHIBA_E/GAIT_Dataset/CASIA-B-pkl/'

# label_set = [f'{i:03d}' for i in range(75, 125)]
# data_in_use = None
# i = 0
# def get_seqs_info_list(label_set):
#             seqs_info_list = []
#             for lab in label_set:
#                 for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
#                     i = 0
#                     for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
#                         seq_info = [lab, typ, vie] # ['001', 'bg-01', '000']
#                         seq_path = osp.join(dataset_root, *seq_info) # /media/ani/TOSHIBA_E/GAIT_Dataset/CASIA-B-pkl/001/bg-01/000
#                         seq_dirs = sorted(os.listdir(seq_path)) #['000.pkl']
#                         if seq_dirs != []:
#                             seq_dirs = [osp.join(seq_path, dir) for dir in seq_dirs]
#                             seqs_info_list.append([*seq_info, seq_dirs])
#                         else:
#                             print('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
#                         i = i + 1
#                     #     break
#                     if(i<11):
#                         print('less views in %s-%s-%s.' % (lab, typ, vie))
                        
#             return seqs_info_list

# def check_subdirectories(seq_info_list, label_set):
#     for seq_info in seq_info_list:
#         label, typ, vie, seq_dirs = seq_info

#         all_pkl_files = set()
#         for dir_path in seq_dirs:
#             pkl_files = {file for file in os.listdir(dir_path) if file.endswith('.pkl')}
#             all_pkl_files.update(pkl_files)

#         if set(label_set) == all_pkl_files:
#             print(f"All subdirectories of {label}-{typ}-{vie} have the same .pkl files as label_set.")
#         else:
#             print(f"Not all subdirectories of {label}-{typ}-{vie} have the same .pkl files as label_set.")


# seq = get_seqs_info_list(label_set)

# # check_subdirectories(seq, label_set)

# import pickle
# import matplotlib.pyplot as plt

# with open('/media/ani/TOSHIBA_E/GAIT_Dataset/CASIA-B-pkl/054/nm-01/036/036.pkl', 'rb') as f:
#     x = pickle.load(f)
    
# plt.imshow(x[21], cmap='gray')

# import h5py
# filename = "/home/ani/Riya/generative-compression/data/cityscapes_paths_train.h5"


# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     # data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     # ds_arr = f[a_group_key][()]  # returns as a numpy array










































