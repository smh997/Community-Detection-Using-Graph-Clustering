# Community-Detection-Using-Graph-Clustering
> Computing Communities in Large Community Networks using Graph Clustering Algorithms

## Algorithms
- Girvan-Newman
- Louvain 
- Spectral Clustering
- Walktrap

## Datasets
- Zachary Karate Club
- American College Football
- DBLP

### Set up

The Python version we used is `3.10.7`

You only need to install communities and community packages as follows:

```
$ pip install communities
$ pip install python-louvain
```

Then you can simply run it by the command below:

```
python main.py
```

Sample result on Karate Club dataset:
![karate](https://user-images.githubusercontent.com/38455739/232283338-5033bc69-93ba-4759-b2eb-ad88bcc883c2.png)

Sample result on Football dataset:
![football](https://user-images.githubusercontent.com/38455739/232283374-4a4df203-1764-4b33-af38-12caaa46bf07.png)

Sample Modularity vs iteration for Walktrap on DBLP:
![image](https://user-images.githubusercontent.com/38455739/232283567-681506b0-a52e-466f-9ebf-b196ce9d68c3.png)

Sample Quality of partioning metric Eta vs iteration for Walktrap on Football dataset:
![image](https://user-images.githubusercontent.com/38455739/232283600-9247f32e-a238-4053-bcc1-76f47b696b33.png)
