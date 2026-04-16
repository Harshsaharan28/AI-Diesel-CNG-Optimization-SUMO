import xml.etree.ElementTree as ET
import copy

BASE_ROUTES   = "routes2_base.rou.xml"
DIESEL_ROUTES = "routes2_diesel.rou.xml"
CNG_ROUTES    = "routes2_cng.rou.xml"


def load_base():
    tree = ET.parse(BASE_ROUTES)
    root = tree.getroot()
    return root


def write_routes(root, fuel_type, out_file):
    root_copy = copy.deepcopy(root)
    for veh in root_copy.findall("vehicle"):
        veh.set("type", fuel_type)
    tree_out = ET.ElementTree(root_copy)
    tree_out.write(out_file, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote {out_file} with type='{fuel_type}'")


def main():
    root = load_base()
    write_routes(root, "diesel", DIESEL_ROUTES)
    write_routes(root, "cng",    CNG_ROUTES)


if __name__ == "__main__":
    main()