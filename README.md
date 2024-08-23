The Python scripts provided in this repository are used for creating datasets, training models, and applying the trained models. It is important to note that this repository does not include the code for generating printhead data.

The creation of printhead data is a labor-intensive and time-consuming process. It requires printing sample sheets and then scanning them at 9600 dpi resolution using a scanner. Due to the large file sizes and the complexity of the task, each section of the printhead must be scanned individually, as handling the full dataset at once would overwhelm most computers.

After scanning, the printhead samples must be transformed into dot clusters using specific image processing methods. This step often takes significant time due to the limitations of the scanner, as it might incorrectly identify smudges or artifacts as ink dots. In our early work, we manually filtered out these erroneous points and dealt with printhead nozzles that failed to eject ink by generating virtual data based on the corresponding positions.

Once we have the dot clusters, we apply the theoretical model discussed in our paper to convert these clusters into printhead data using certain algorithms.

Currently, we are proposing a more direct method for converting printed sample images into printhead data. However, as the related paper is still under review and has not been posted as a preprint, we are unable to provide a link to it at this time. Once the paper is published, we will update this repository with the link.

Regarding the printhead data, due to the complexity of processing, we are only using data from the first two printhead groups. For the remaining groups, we substitute the data with idealized printhead data. As a result, the gamut covered by the first two groups is accurately modeled, while the quality of the modeling outside this gamut is poorer because the data is synthetic.

Therefore, for our experiments, we only use the first two printhead groups and restrict the image data to regions within their gamut. Within this range, the color regression and fitting perform well.

Here, I use the image of the fruit as the training set. The training set is much smaller than the dataset, proving that the model has strong generalization ability within this color gamut. The links to the image and dataset are provided below:
https://mega.nz/folder/WZh0FKhT#_iFxhbAEXb3jf6SPjzOghQ
