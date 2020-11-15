# Helper script to create processed_tiles.json

def th(leftTerrain, leftCrowns, rightTerrain, rightCrowns):
    tt = ["desert", "forest", "ocean", "grass", "swamp", "mine"]
    assert (leftTerrain in tt )
    assert (rightTerrain in tt )
    assert (leftCrowns in [0, 1, 2, 3])
    assert (rightCrowns in [0, 1, 2, 3])
    return {
        "left_terrain": leftTerrain,
        "left_crowns": leftCrowns,
        "right_terrain": rightTerrain,
        "right_crowns": rightCrowns
    }

# These are manually created labels
whole_tiles = {
    "01": th("desert", 0, "desert", 0),
    "02": th("desert", 0, "desert", 0),
    "03": th("forest", 0, "forest", 0),
    "04": th("forest", 0, "forest", 0),
    "05": th("forest", 0, "forest", 0),
    "06": th("forest", 0, "forest", 0),
    "07": th("ocean", 0, "ocean", 0),
    "08": th("ocean", 0, "ocean", 0),
    "09": th("ocean", 0, "ocean", 0),
    "10": th("grass", 0, "grass", 0),
    "11": th("grass", 0, "grass", 0),
    "12": th("swamp", 0, "swamp", 0),
    "13": th("desert", 0, "forest", 0),
    "14": th("desert", 0, "ocean", 0),
    "15": th("desert", 0, "grass", 0),
    "16": th("desert", 0, "swamp", 0),
    "17": th("forest", 0, "ocean", 0),
    "18": th("forest", 0, "grass", 0),
    "19": th("desert", 1, "forest", 0),
    "20": th("desert", 1, "ocean", 0),
    "21": th("desert", 1, "grass", 0),
    "22": th("desert", 1, "swamp", 0),
    "23": th("desert", 1, "mine", 0),
    "24": th("forest", 1, "desert", 0),
    "25": th("forest", 1, "desert", 0),
    "26": th("forest", 1, "desert", 0),
    "27": th("forest", 1, "desert", 0),
    "28": th("forest", 1, "ocean", 0),
    "29": th("forest", 1, "grass", 0),
    "30": th("ocean", 1, "desert", 0),
    "31": th("ocean", 1, "desert", 0),
    "32": th("ocean", 1, "forest", 0),
    "33": th("ocean", 1, "forest", 0),
    "34": th("ocean", 1, "forest", 0),
    "35": th("ocean", 1, "forest", 0),
    "36": th("desert", 0, "grass", 1),
    "37": th("ocean", 0, "grass", 1),
    "38": th("desert", 0, "swamp", 1),
    "39": th("grass", 0, "swamp", 1),
    "40": th("mine", 1, "desert", 0),
    "41": th("desert", 0, "grass", 2),
    "42": th("ocean", 0, "grass", 2),
    "43": th("desert", 0, "swamp", 2),
    "44": th("grass", 0, "swamp", 2),
    "45": th("mine", 2, "desert", 0),
    "46": th("swamp", 0, "mine", 2),
    "47": th("swamp", 0, "mine", 2),
    "48": th("desert", 0, "mine", 3)
}

if __name__ == "__main__":
    prefix = "processed_tiles/"
    for i in range(1, 49):
        filename = prefix + "cropped_tile_"
        
        key = str(i)
        if (i < 10): key = "0" + key
        
        filename += key + ".png"    
        whole_tiles[key]["image_path"] = filename

    import json
    with open("processed_tiles.json", "w") as outfile:
        json.dump(whole_tiles, outfile)
