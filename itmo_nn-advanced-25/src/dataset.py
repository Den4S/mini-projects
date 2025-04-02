import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# INIT_MASK_SIZE = 64
# CENTERED_MASK_SIZE = INIT_MASK_SIZE
# CUTTED_MIN_WIDTH = 15
# BRIGHTNESS_LIM = 0.05


class WeizmannSeqs(Dataset):

    def __init__(
        self,
        ds_filepathes,
        seq_len: int,
        skip: int,
        transformations: transforms.Compose | None = None,
        augmentation_horizontally_flip: bool = False,
        exclude_cutted: bool = False,
        centering: bool = False,
    ):
        """
        Parameters
        ----------
        ds_filepathes : list
            A list of filepathes with masks sequences.
        seq_len: int
            Number of frames in each sub-sequence in which videos are devided.
        skip: int
            A frame interval for each sub-sequence.
        transformations : transforms.Compose
            A sequence of transforms that will be applied to each frame.

        augmentation_horizontally_flip: bool
            If True an augmentation (horizontally flip) applying.
            Size of the dataset increases 2 times! 
        exclude_cutted: bool
            A flag that is responsible for exluding of 'not complete' silhouettes!
        centering: bool
            A flag that is responsible for silhouettes centering on the frame!
        """
        self.ds_filepathes = ds_filepathes
        self.seq_len = seq_len
        self.skip = skip
        
        self.augmentation_horizontally_flip = augmentation_horizontally_flip
        self.exclude_cutted = exclude_cutted
        self.centering = centering
        
        # load all masks at once
        self.all_np_masks, self.files_lengths, self.files_silhouettes_coord = self.load_masks()
        self.ds_constructor, self.seqs_counter = self.get_ds_constructor()
        
        self.transformations = transformations

    def check_silhouette_not_cutted(self, filepath, frame_ind):
        """
        Returns a boolean value telling if a silhouette is cutted (incomplete).
            True  - silhouette is OK
            False - silhouette is cutted, it is better to exclude it
        """
        if self.files_silhouettes_coord[filepath][frame_ind] is None:
            return False  # no silhouette!

        upper_left_y, upper_left_x, lower_right_y, lower_right_x = self.files_silhouettes_coord[filepath][frame_ind]

        if (upper_left_x == 0) or (lower_right_x == INIT_MASK_SIZE - 1):
            # 1. one of boundaries of a silhouette bounding box coincides with a boundary of a frame;
            width = lower_right_y - upper_left_y

            if width < CUTTED_MIN_WIDTH:
                # 2. if 1^st criteria satisfied and a width of a silhouette bounding box is less than some value
                return False
        # all is OK
        return True

    def get_ds_constructor(self):
        """
        Returns
            1. list of pairs (filename index, [timesteps for sequence]);
            2. dictianory {action: number of sequences in dataset of the certain kind}.

        Comment: Here we also exclude all subsequences with cutted silhouette (if self.exclude_cutted is True)
        """
        ds_constructor = []  # pairs: (train file index, [self.seq_len timesteps in the file])
        seqs_counter = {}
        
        for ind_file, filepath in enumerate(self.ds_filepathes):
            action_name = filepath.split('/')[-2].split('_')[0]  # depends on a data directory!
            if action_name not in seqs_counter.keys():
                seqs_counter[action_name] = 0
            
            timesteps_this = self.files_lengths[filepath]
            opposite_boundary_reached = False
            
            for ind_timestep in range(timesteps_this):
                if self.exclude_cutted: # check if not cutted
                    if self.check_silhouette_not_cutted(filepath, ind_timestep):
                        pass
                    else:  # first subseq silhouette is cutted - skip
                        continue

                seq_this = [ind_timestep]
                
                ind_this = ind_timestep
                for step in range(1, self.seq_len):  # `self.seq_len` frames for each sub-sequence 
                    ind_this += self.skip + 1  # with a frame interval of `skip`
                    if ind_this <= timesteps_this - 1:
                        if self.exclude_cutted: # check if not cutted (reached another boundary of a frame!)
                            if self.check_silhouette_not_cutted(filepath, ind_this):
                                pass
                            else:  # first subseq silhouette is cutted - skip
                                opposite_boundary_reached = True
                                break
                        seq_this.append(ind_this)

                if self.exclude_cutted and opposite_boundary_reached:
                    break  # go to the next file! opposite boundary reached!
            
                if len(seq_this) == self.seq_len:
                    ds_constructor.append((ind_file, seq_this))
                    if not self.augmentation_horizontally_flip:
                        seqs_counter[action_name] += 1
                    else:  # count each sequence twice (normal + flipped)
                        seqs_counter[action_name] += 2

        return ds_constructor, seqs_counter

    def load_masks(self):
        """
        Load all masks by filepathes and returns two dictionaries
            1. {filepath: video masks}
            2. {filepath: number of frames in a video}
            3. {filepath: list (len = number of timesteps in a file) of tuples;
                          each tuple: (ul_corner_y, ul_corner_x, lr_corner_y, lr_corner_x)}
        """
        all_np_masks = {}
        files_lengths = {}
        files_silhouette_bounds = {}

        for filepath in self.ds_filepathes:
            all_np_masks[filepath] = np.load(filepath)  # [:, 0, :, :]
            files_lengths[filepath] = all_np_masks[filepath].shape[0]

            if self.centering or self.exclude_cutted:  # if we need to center each silhouette
                # for each timestep of a video finding a bounding box for a silhouette
                for ind_frame in range(files_lengths[filepath]):
                    frame_this = all_np_masks[filepath][ind_frame, 0, :, :]

                    strickt_silhouette = torch.where(
                        torch.tensor(frame_this) > BRIGHTNESS_LIM, 1.0, 0.0
                    )

                    if strickt_silhouette.sum() > 0:
                        # indices, where a silhouette is:
                        y_ids = torch.where(strickt_silhouette.sum(dim=1) > 0)[0]  # along y direction
                        x_ids = torch.where(strickt_silhouette.sum(dim=0) > 0)[0]  # along x direction

                        coordinates_this = (
                            y_ids[0].item(), x_ids[0].item(),
                            y_ids[-1].item(), x_ids[-1].item()
                        )
                    else:  # no silhouette (is it possible?)
                        coordinates_this = None

                    if filepath in files_silhouette_bounds.keys():
                        files_silhouette_bounds[filepath].append(coordinates_this)
                    else:
                        files_silhouette_bounds[filepath] = [coordinates_this]

        return all_np_masks, files_lengths, files_silhouette_bounds

    def __len__(self):
        if self.augmentation_horizontally_flip:
            return 2 * len(self.ds_constructor)
        else:
            return len(self.ds_constructor)

    def centered_silhouette(self, filepath, frame_ind):
        """
        Returns centered silhouettes for frame number `frame_ind` of `filepath` video.
        """
        # silhouette box coordinates
        upper_left_y, upper_left_x, lower_right_y, lower_right_x = self.files_silhouettes_coord[filepath][frame_ind]
        
        silhouette_this = torch.tensor(
            self.all_np_masks[filepath][frame_ind, 0, upper_left_y:lower_right_y + 1, upper_left_x:lower_right_x + 1]
        )  # size equal to bounding box size!
        rect_y, rect_x = silhouette_this.shape
        
        # transform selected rectangle to square, centering a silhouette
        square_size = CENTERED_MASK_SIZE
        # paddings along OY
        pad_top_silhouette = int((square_size - rect_y) / 2)
        pad_bottom_silhouette = square_size - pad_top_silhouette - rect_y
        # paddings along OX
        pad_left_silhouette = int((square_size - rect_x) / 2)
        pad_right_silhouette = square_size - pad_left_silhouette - rect_x  # params for transforms.Pad
        
        transform_for_silhouette = transforms.Pad(
            padding=(
                pad_left_silhouette,  # left padding
                pad_top_silhouette,  # top padding
                pad_right_silhouette,  # right padding
                pad_bottom_silhouette  # bottom padding
            ),
            fill=0,
        )  # 0's padding to match sizes!

        return transform_for_silhouette(silhouette_this).unsqueeze(0)

    def __getitem__(self, ind: int) -> tuple:
        """
        Returns tensor for a frame sequence and its label (action index).
        Comment: Here we are also centering silhouettes if self.centering == True.
        
        Parameters
        ----------
        ind : int
            Index of element to return.

        Returns
        -------
        tuple
            An element of dataset: tuple(torch.Tensor, class)
        """
        if self.augmentation_horizontally_flip:
            ind_this = ind // 2  # if ind is odd - normal video, if ind is even - horizontally flipped
        else:
            ind_this = ind
    
        ind_filepath, frames_ids = self.ds_constructor[ind_this]
        
        filepath = self.ds_filepathes[ind_filepath]
        action_name = filepath.split('/')[-2].split('_')[0]  # depends on a data directory!
        label = ACTION_TO_ID[action_name]  # label for the sequence

        if (not self.augmentation_horizontally_flip) or (self.augmentation_horizontally_flip and ind_this % 2 == 0):
            sequence_raw = [
                torch.tensor(
                    self.all_np_masks[filepath][frame_ind, :, :, :]  # preserving dim=1 - channels
                )
                if not self.centering else
                self.centered_silhouette(filepath, frame_ind)
                for frame_ind in frames_ids
            ]
        else:  # horizontally flipped frames
            sequence_raw = [ 
                torch.flip(  # horizontally flip
                    torch.tensor(
                        self.all_np_masks[filepath][frame_ind, :, :, :]  # preserving dim=1 - channels
                    ), dims=[-1]
                ) 
                if not self.centering else
                torch.flip(self.centered_silhouette(filepath, frame_ind), dims=[-1])
                for frame_ind in frames_ids
            ]
        
        # apply transformations
        if self.transformations:
            sequence_item = torch.stack(
                [self.transformations(frame) for frame in sequence_raw], dim=0
            )
        else:
            sequence_item = sequence_raw

        return sequence_item, label
