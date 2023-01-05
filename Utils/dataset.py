from typing import List
import torch
from torch.utils.data import Dataset
import os
import gc
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(
        self,
        folder_path_list: List[str],
        response_type="Acceleration",
        numOfData_per_folder=None,
    ):
        self.folder_path_list = folder_path_list  # [ChiChi_path, NGAWest2_path]
        self.numOfData_per_folder = numOfData_per_folder
        self.unified_sec = 100
        self.unified_sample_rate = 0.005
        self.response_type = response_type
        self.source_normalization_state = False
        self.target_normalization_state = False
        self.response_list = [
            "Acceleration",
            "Velocity",
            "Displacement",
            "Moment_Z",
            "Shear_Y",
        ]
        self.graphs = self.__load()  # List

    def __load(self):
        graphs = []
        graph_name = "structure_graph.pt"
        for folder_path in self.folder_path_list:
            data_list = sorted(
                os.listdir(folder_path),
                key=lambda data_name: int(data_name.split("_")[-1]),
            )
            numOfData = (
                len(data_list)
                if self.numOfData_per_folder is None
                else min(len(data_list), self.numOfData_per_folder)
            )
            print(f"Loading data from {folder_path}")
            for i in tqdm(range(numOfData)):
                graph = torch.load(os.path.join(folder_path, data_list[i], graph_name))

                # discord the data which's sample_rate != unified_sample_rate:
                if graph.sample_rate != self.unified_sample_rate:
                    continue

                # 1. pad or cut ground motion to 100 sec
                unified_seq_len = int(self.unified_sec / graph.sample_rate)
                # cut
                if graph.time_steps > unified_seq_len:
                    graph.time_steps = unified_seq_len
                    graph.ground_motion = graph.ground_motion[:unified_seq_len]
                # pad
                else:
                    target_len = unified_seq_len - graph.time_steps
                    padded_seq = torch.zeros(target_len)
                    graph.ground_motion = torch.cat(
                        [graph.ground_motion, padded_seq], dim=0
                    )

                # 2. delete useless response
                graph.response_type = self.response_type
                graph.story = int(graph.x[0, 1]) - 1
                graph.y = graph[self.response_type]
                for response_type in self.response_list:
                    del graph[response_type]
                    gc.collect()

                # 3. expand one dim for data concatenation
                graph.ground_motion = torch.unsqueeze(graph.ground_motion, dim=0)
                # 因為原本用 numpy array 轉成 tensor 傳入的 dtype 是 torch.float64, RNN 只吃 float
                graph.ground_motion = graph.ground_motion.float()
                if (
                    graph.response_type == "Acceleration"
                    or graph.response_type == "Velocity"
                    or graph.response_type == "Displacement"
                ):
                    graph.y = torch.unsqueeze(graph.y, dim=0)

                graphs.append(graph)

        print(f"\033[0;32;49m {'=' * 100 } \033[0;0m")
        print(f"\033[0;32;49m  number of effective data: {len(graphs)} \033[0;0m")
        print(f"\033[0;32;49m {'=' * 100 } \033[0;0m")
        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]

    @property
    def source_norm_state(self):
        return self.source_normalization_state

    @property
    def target_norm_state(self):
        return self.target_normalization_state

    def get_normalized_item_dict(self):
        normalized_item_dict = {}
        x = {}
        edge_attr = {}
        if not self.source_normalization_state and not self.target_normalization_state:
            x["XYZ_gridline_num"] = max(
                [torch.max(torch.abs(data.x[:, :3])) for data in self.graphs]
            )
            x["XYZ_grid_index"] = max(
                [torch.max(torch.abs(data.x[:, 3:6])) for data in self.graphs]
            )
            x["period"] = max(
                [torch.max(torch.abs(data.x[:, 6])) for data in self.graphs]
            )
            x["DOF"] = max([torch.max(torch.abs(data.x[:, 7])) for data in self.graphs])
            x["mass"] = max(
                [torch.max(torch.abs(data.x[:, 8])) for data in self.graphs]
            )
            x["XYZ_inertia"] = max(
                [torch.max(torch.abs(data.x[:, 9:12])) for data in self.graphs]
            )
            x["XYZ_mode_shape"] = max(
                [torch.max(torch.abs(data.x[:, 12:15])) for data in self.graphs]
            )
            edge_attr["S_y"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 2])) for data in self.graphs]
            )
            edge_attr["S_z"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 3])) for data in self.graphs]
            )
            edge_attr["area"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 4])) for data in self.graphs]
            )
            edge_attr["element_length"] = max(
                [torch.max(torch.abs(data.edge_attr[:, 5])) for data in self.graphs]
            )
            normalized_item_dict["x"] = x
            normalized_item_dict["ground_motion"] = max(
                [torch.max(torch.abs(data.ground_motion)) for data in self.graphs]
            )
            normalized_item_dict["y"] = max(
                [torch.max(torch.abs(data.y)) for data in self.graphs]
            )
            normalized_item_dict["edge_attr"] = edge_attr
            normalized_item_dict["response_type"] = self.response_type
        return normalized_item_dict

    def normalize_source(self, normalized_item_dict):
        if self.source_normalization_state is False:
            # normalize
            for data in self.graphs:
                data.x[:, :3] = (
                    data.x[:, :3] / normalized_item_dict["x"]["XYZ_gridline_num"]
                )
                data.x[:, 3:6] = (
                    data.x[:, 3:6] / normalized_item_dict["x"]["XYZ_grid_index"]
                )
                data.x[:, 6] = data.x[:, 6] / normalized_item_dict["x"]["period"]
                data.x[:, 7] = data.x[:, 7] / normalized_item_dict["x"]["DOF"]
                data.x[:, 8] = data.x[:, 8] / normalized_item_dict["x"]["mass"]
                data.x[:, 9:12] = (
                    data.x[:, 9:12] / normalized_item_dict["x"]["XYZ_inertia"]
                )
                data.x[:, 12:15] = (
                    data.x[:, 12:15] / normalized_item_dict["x"]["XYZ_mode_shape"]
                )
                data.edge_attr[:, 2] = (
                    data.edge_attr[:, 2] / normalized_item_dict["edge_attr"]["S_y"]
                )
                data.edge_attr[:, 3] = (
                    data.edge_attr[:, 3] / normalized_item_dict["edge_attr"]["S_z"]
                )
                data.edge_attr[:, 4] = (
                    data.edge_attr[:, 4] / normalized_item_dict["edge_attr"]["area"]
                )
                data.edge_attr[:, 5] = (
                    data.edge_attr[:, 5]
                    / normalized_item_dict["edge_attr"]["element_length"]
                )
                data.ground_motion = (
                    data.ground_motion / normalized_item_dict["ground_motion"]
                )
            # change source normalization state
            self.source_normalization_state = True

    def normalize_target(self, normalized_item_dict):
        if self.target_normalization_state is False:
            # normalize
            for data in self.graphs:
                data.y = data.y / normalized_item_dict["y"]
            # change target normalization state
            self.target_normalization_state = True

    def denormalize_source(self, normalized_item_dict):
        if self.source_normalization_state is True:
            # denormalize
            for data in self.graphs:
                data.x[:, :3] = (
                    data.x[:, :3] * normalized_item_dict["x"]["XYZ_gridline_num"]
                )
                data.x[:, 3:6] = (
                    data.x[:, 3:6] * normalized_item_dict["x"]["XYZ_grid_index"]
                )
                data.x[:, 6] = data.x[:, 6] * normalized_item_dict["x"]["period"]
                data.x[:, 7] = data.x[:, 7] * normalized_item_dict["x"]["DOF"]
                data.x[:, 8] = data.x[:, 8] * normalized_item_dict["x"]["mass"]
                data.x[:, 9:12] = (
                    data.x[:, 9:12] * normalized_item_dict["x"]["XYZ_inertia"]
                )
                data.x[:, 12:15] = (
                    data.x[:, 12:15] * normalized_item_dict["x"]["XYZ_mode_shape"]
                )
                data.edge_attr[:, 2] = (
                    data.edge_attr[:, 2] * normalized_item_dict["edge_attr"]["S_y"]
                )
                data.edge_attr[:, 3] = (
                    data.edge_attr[:, 3] * normalized_item_dict["edge_attr"]["S_z"]
                )
                data.edge_attr[:, 4] = (
                    data.edge_attr[:, 4] * normalized_item_dict["edge_attr"]["area"]
                )
                data.edge_attr[:, 5] = (
                    data.edge_attr[:, 5]
                    * normalized_item_dict["edge_attr"]["element_length"]
                )
                data.ground_motion = (
                    data.ground_motion * normalized_item_dict["ground_motion"]
                )
            # change source normalization state
            self.source_normalization_state = False

    def denormalize_target(self, normalized_item_dict):
        if self.target_normalization_state is True:
            # denormalize
            for data in self.graphs:
                data.y = data.y * normalized_item_dict["y"]
            # change target normalization state
            self.target_normalization_state = False
