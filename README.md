# Machine Learning in Medicine - Theory & Practice

The course materials contained in this repository were part of the XXXXX paper link XXXXX, created jointly by the Faculty of Mathematics at the University Duisburg-Essen and the Institute of AI in Medicine (IKIM). They should work out of the box, if you do the following:
- 1) Copy the entirety of the learning materials to any machine that has GPUs.
- 2) Use anaconda or miniconda to create an environment. The environment.yaml file should be used as input here.
- 3) To acquire the data, we recommend downloading the PNG images of the LiTS 2017 dataset by visiting https://www.kaggle.com/datasets/andrewmvd/lits-png. If you want to recreate the structure of the files we used, misc/ contains the train/val/test split that our students worked with as a tree view. The misc/ folder also contains the respective classes.csv files you can see in the tree view, which contain the solutions that the training relies on.
- 4) If you are not using the cloud platform we designed (https://github.com/TIO-IKIM/coder-aws), the paths under which the environments and data are expected are different from the ones we put into the files. In this case, you will have to manually edit every file which uses some sort of explicit sys.path.append or something like it.

This seminar is intended for people with some prior experience in Python or some other programming language and at least cursory knowledge of Machine Learning. Neither is strictly necessary, but those who have never programmed before will experience a steep learning curve past the initial tutorial. We had people with little or no prior experience in our first seminar, and they did manage to pass with good grades, given some help by tutors and their team members.

If you have any questions, suggestions, complaints, issues, or comments, please file a GitHub issue. We intend to address them all. If we do not, feel free to send an E-Mail to me at frederic.jonske@uk-essen.de.