from utils.geocoder import extract_location, geocode_location

def test_geocoder():
    # Test cases
    test_cases = [
        ("Massive fire reported in a Mumbai residential building.", "mumbai"),
        ("New expressway project inaugurated in Delhi today.", "delhi"),
        ("Heavy rains cause flooding in Chennai suburbs.", "chennai"),
        ("Tech summit to be held in Bengaluru next month.", "bengaluru"),
        ("No location mentioned in this headline.", None),
        ("Police arrested a gang in Navi Mumbai for gold smuggling.", "navi mumbai"),
    ]

    for text, expected in test_cases:
        loc = extract_location(text)
        print(f"Text: '{text}' -> Extracted: {loc}")
        assert loc == expected
        
        if loc:
            lat, lon = geocode_location(loc)
            print(f"  Coordinates: {lat}, {lon}")
            assert lat is not None and lon is not None

    print("\nAll geocoder tests passed!")

if __name__ == "__main__":
    test_geocoder()
