class StrokeDataset(Dataset):
    def __init__(self, images, pixels):
        self.images = torch.tensor(images, dtype=torch.int32)
        self.pixels = torch.tensor(pixels, dtype=torch.float32)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        img = self.images[index]
        px = self.pixels[index]
        return img, px


def get_omniglot_dataset(dataset_path):

    canvas_data = []
    pixel_data = []
    for character_data in dataset:
        for stroke_sample in character_data['motor']:
            sequence = get_sequence(stroke_sample)
            canvas_list, pixel_list = process_training_samples(sequence, canvas_size)
            canvas_data += canvas_list
            pixel_data += pixel_list

    canvas_data = np.array(canvas_data)
    pixel_data = np.array(pixel_data)

    dataset = StrokeDataset(images=canvas_data, pixels=pixel_data)

def main():
    dataset = get_omniglot_dataset(dataset_path)

if __name__ == '__main__':
    main()
