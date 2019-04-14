


if __name__ == '__main__':
  self.data = pd.read_csv(path_to_file)
  self.data['filename'] = path_to_data + self.data['filename']
  self.unique_images = self.data['filename'].unique()