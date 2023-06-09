import torch


class Finder:
    

    def find(self, datasource, save_path=None):
        raise NotImplementedError()
    
    @classmethod
    def find_images(cls, model1, model2, number, test_loader, agreement, device="cpu"):
        """返回number张在test_loader中，model1和model2分类一致的图片。

        Args:
            model1 (torch.nn.Module): _description_
            model2 (torch.nn.Module): _description_
            number (int): _description_
            test_loader (torch.utils.data.DataLoader): _description_

        Returns:
            Tuple(Tensor, Tensor): _description_
        """
        same_indices = []
        same_images = []
        same_labels = []
        for i, (images, label) in enumerate(test_loader):
            images = images.to(device)
            label  = label.to(device)
            pred_m = torch.argmax(model1(images), dim=-1);
            pred_t = torch.argmax(model2(images), dim=-1);
            # print(pred_m- pred_t)
            if agreement:
                indices = torch.where(pred_m == pred_t)[0]
            else:
                indices = torch.where(pred_m != pred_t)[0]
            # indices += i * 128
            same_indices.append(indices)
            same_images.append(images[indices, :, :, :])
            same_labels.append(label[indices])
            # break
        same_images = torch.cat(same_images)
        same_labels = torch.cat(same_labels)
        return same_images[: number], same_labels[: number]

