import sys
import json
import cv2
import numpy as np
from tensorflow import keras

# This code is pieced together from 
# the exploratoy work in ParseTilesScratch.ipynb

terrainToInt = {
    "desert": 0,
    "forest": 1,
    "ocean": 2,
    "grass": 3,
    "swamp": 4,
    "mine": 5
}


class FullTiles:
    def __init__(self):
        try:
            with open('processed_tiles.json') as json_file:
                self.whole_tiles = json.load(json_file)
                self.loadImages()

        except (FileNotFoundError, cv2.error):
            print("Unable to load json file containing tile information.")
            sys.exit()

    def loadImages(self):
        for key, tile in self.whole_tiles.items():
            tileimg = cv2.cvtColor(cv2.imread(tile["image_path"]), cv2.COLOR_BGR2RGB)
            self.whole_tiles[key]["image"] = tileimg
            self.whole_tiles[key]["intensity"] =  cv2.cvtColor(tileimg, cv2.COLOR_RGB2GRAY)


class HalfTiles:
    def __init__(self, full):
        # Convert full-tile representation to half-tile
        self.tiles = {}
        for key, tile in full.whole_tiles.items():
            t = (key, True)
            self.tiles[t] = {
                "parent": key,
                "terrain": tile["left_terrain"],
                "terraini": terrainToInt[tile["left_terrain"]],
                "crowns": tile["left_crowns"],
                "image": tile["image"][:, 0:128].copy(),
                "intensity": tile["intensity"][:, 0:128].copy(),
                "family": tile["family"]
            }
            t = (key, False)
            self.tiles[t] = {
                "parent": key,
                "terrain": tile["right_terrain"],
                "terraini": terrainToInt[tile["right_terrain"]],
                "crowns": tile["right_crowns"],
                "image": tile["image"][:, 128:].copy(),
                "intensity": tile["intensity"][:, 128:].copy(),
                "family": tile["family"]
            }
        
        # Load crown template
        crown = cv2.imread("processed_tiles/cropped_crown.png")
        crown = cv2.cvtColor(crown, cv2.COLOR_BGR2RGB)
        self.crown = crown / 255.  
        crown_mask = cv2.imread("processed_tiles/cropped_crown_mask.png").astype('bool').astype('uint8')[:,:,0]
        self.crown_mask = crown_mask / 1.

        # Load models
        self.terrain_model = keras.models.load_model('models/terrain_model.keras')
        self.crowns_model = keras.models.load_model('models/crowns_model.keras')

    def predictTerrain_old(self, image):
        """Returns a predicted terrain type for the given 128x128 image"""
        assert(image.shape[0:2] == (128, 128))
        ccenters = np.array([[161.48790095, 141.80672983, 10.09729473],
                    [85.94123424,  93.49039529,  41.31753817],
                    [74.23682319, 107.92738851, 138.87711249],
                    [117.44718715, 137.31491961,  30.39860317],
                    [124.87008057, 110.4710083,   65.12401123],
                    [ 94.07957967,  81.18944295,  49.85673014]])
        avg = np.mean(np.mean(image, axis = 0), axis = 0)
        distance = np.sum((avg - ccenters)**2, axis = 1)
        guess = np.where(distance == np.min(distance))[0][0]
        return guess
#        return {
#            "cluster_center": ccenters[guess],
#            "avg": avg,
#            "distance": distance,
#            "guess": guess
#        }

    def predictTerrain(self, image):
        image = cv2.resize(image, (24, 24))
        image = np.expand_dims(image, axis = 0)
        p = self.terrain_model.predict(image)[0]
        guess = int(np.where(p == np.max(p))[0])
        return guess
    
    def crownLoss(self, image):
        """Internal Function for predictCrowns.  Returns SSD loss of crown template against
             given image.  Taken from my MP2 code."""
        total = np.zeros(image.shape[0:2])
        for c in range(3):
            I = image[:,:,c] / 255.
            T = self.crown[:,:,c]
            M = self.crown_mask
            c = np.sum((M*T)**2)
            d = cv2.filter2D(I, ddepth = -1, kernel = M*T) 
            e = cv2.filter2D(I**2, ddepth = -1, kernel = M)
            r = c - 2.0*d + e
            assert(r.shape == I.shape)
            total = total + r
        return total

    def findCrowns(self, image, thresh = 30, radius = 25):
        """Returns a list [(y, x, L), ...] of coordinates and associated loss of likely locations
        of crowns in the given image.

        thresh=30 is tuned to 128x128 may need adjustment for other sizes.
        radius=25 similarly tuned is used to eliminate candidate points that are close together.
        """
        loss = self.crownLoss(image)
        idx = np.where(loss <= thresh)

        # Remove idx that are close together. Use the lowest loss.
        def helper(ys, xs):
            clusters = []
            for y, x in zip(ys, xs):
                new_point = True
                # Will we add this to our list of candidate crowns?
                # calculate distance to other found centers.
                for i, (yr, xr, closs) in enumerate(clusters):
                    distance = (yr - y)*(yr - y) + (xr - x)*(xr - x)
                    if distance < 25:
                        new_point = False                
                        if loss[y, x] < closs:
                            # This new point is close and has a better loss, so replace it.
                            clusters[i] = (y, x, loss[y, x])

                # If new_point is still tru
                if new_point:
                    clusters.append((y, x, loss[y, x]))
            return clusters
    
        clusters = helper(idx[0], idx[1])
        
        newys = [ c[0] for c in clusters ] 
        newxs = [ c[1] for c in clusters ]
        
        return helper(newys, newxs)

    def predictCrowns_Old(self, image):
        """Returns the predicted number of crowns in an image."""
        assert(image.shape[0:2] == (128, 128))

        N = self.findCrowns(image)
        E = self.findCrowns(np.rot90(image))
        S = self.findCrowns(np.rot90(image, 2))
        W = self.findCrowns(np.rot90(image, 3))
        return np.max([len(N), len(E), len(S), len(W)])

    def predictCrowns(self, image):
        image = cv2.resize(image, (24, 24))
        image = np.expand_dims(image, axis = 0)
        p = self.crowns_model.predict(image)[0]
        guess = int(np.where(p == np.max(p))[0])
        return guess

    def tileCandidates(self, image):
        """Return list of likely half-tile candidates this image could belong too."""
        terrain = self.predictTerrain(image)
        crowns = self.predictCrowns(image)
        return self.tileCandidates2(terrain, crowns)

    def tileCandidates2(self, terrain, crowns):
        result = set([ key for key, tile in self.tiles.items() \
            if tile["terraini"] == terrain and tile["crowns"] == crowns ])
        return result

    def tileCandidatesWithPair(self, terrainSelf, crownsSelf, neighborTerrain, neighborCrowns):
        # find two half tiles with same parent that match on these attributes
        left = self.tileCandidates2(terrainSelf, crownsSelf)
        results = set([])
        for L in left:
            key = (L[0], not L[1])
            p = self.tiles[key]
            if p["terraini"] == neighborTerrain and p["crowns"] == neighborCrowns:
                results.add(key)            
        return results

    def selfTest(self):
        """Runs ground-truth images through predictions and verifies result matches hand-labels."""
        errors = 0
        matches = 0
        for key, tile in self.tiles.items():
            image = tile["image"]
            terrain = self.predictTerrain(image)
            crowns = self.predictCrowns(image)

            terrain_mismatch = (terrain != tile["terraini"])
            crown_mismatch = (crowns != tile["crowns"])

            if terrain_mismatch or crown_mismatch:
                print("Mismatch in tile {0}.  Actual (Terrain/Crowns) {1}/{2}.  Predicted {3}/{4}".format(
                    key, tile["terrain"], tile["crowns"], terrain, crowns
                ))
                errors = errors + 1
            else:
                potential_matches = self.tileCandidates(image)
                print("Match in tile {0}.  Potential exact matches {1}".format(key, potential_matches))
                matches = matches + 1

        print("Total Errors {0}.\nTotal Matches {1}.\n".format(errors, matches))
        if errors > 0:
            raise ValueError
            
if __name__ == "__main__":
    tiles = HalfTiles(FullTiles())
    tiles.selfTest()

