from fastai.vision.all import L, Path, PILImage, LabeledBBox, TensorBBox, load_image
from fastcore.test import *
from icevision.all import *
import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime
from pprint import pprint
import sys, os, pickle





# -----------------------------------------------------------------------------
# -----
# -----     Paths
# -----


# high-level paths
BASE_PATH = Path("/home/rory/repos/trailcam")
DATA = Path("/home/rory/data")
MODELS = Path("/home/rory/models/trailcam")


# specific image data sources & annos
TRAILCAM = DATA / "trailcam"
TRAILCAM_ANNOS = TRAILCAM / "annos.json" # renamed from TEST

NACTI = DATA / "nacti"
NACTI_ANNOS = NACTI / "metadata.json"

COCO = DATA / "lvis"
COCO_TRAIN = COCO / "lvis_v1_train.json"
COCO_VALID = COCO / "lvis_v1_val.json"
COCO_FLAT  = COCO / "projects" / "lvis_v1_train-flat.json"
COCO_ID2CAT = MODELS / "id2cat.pickle" # renamed from ID2CAT

BING = DATA / "bing"






# -----------------------------------------------------------------------------
# -----
# -----     Variables
# -----


# Misc
turkey_id = 9999


# Dataloaders
im_sz   = 128*3
pre_sz  = 128*4
bs      = 32
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=im_sz, presize=pre_sz),
                             tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(im_sz),
                             tfms.A.Normalize()])

# Model
arch = models.ross.efficientdet
backbone = arch.backbones.tf_lite0


# Training
lr            = 1e-3
base_lr       = 1e-3
freeze_epochs = 3
epochs        = 10


# Inference
thresh = 0.4


        
        


# -----------------------------------------------------------------------------
# -----
# -----     Model
# -----


def get_model(cats):
    model = arch.model(
        backbone=backbone(pretrained=True),
        num_classes=len(ClassMap(cats)),
        img_size=im_sz)
    return model



def train_model(model, dls, save_path=None, lr=None, freeze_epochs=freeze_epochs, epochs=epochs):
    learn = arch.fastai.learner(
        dls=dls,
        model=model,
        metrics=[COCOMetric(metric_type=COCOMetricType.bbox)])
    if lr == None:
        print("Running learn.lr_find() ...")
        lr = learn.lr_find()
        lr = lr.valley
    print(f"Using lr: {lr}")
    learn.fine_tune(base_lr=lr, freeze_epochs=freeze_epochs, epochs=epochs)
    if save_path != None:
        save_weights(model, save_path)
    return learn






# -----------------------------------------------------------------------------
# -----
# -----     Inference
# -----


def get_infer(weights, cats):
    """Returns a pretrained model in eval mode."""
    infer = get_model(cats)
    infer.load_state_dict(torch.load(weights))
    infer.eval()
    return infer



def get_targs(cats):
    """Gets the appropriate target records for a given list of categories."""
    res = load_json(TRAILCAM_ANNOS)
    res = filter_by_cats(res, cats)
    print(f"Found {len(res)} test records for categories {cats}.")
    return list(res.values())



def pred_to_rec(p):
    return {
        'path': p['path'],
        'width': p['height'],
        'height': p['width'],
        'cats': p['detection']['labels'],
        'bboxes': to_xywh(icebb_to_list(p['detection']['bboxes'])),
        'confs': list(p['detection']['scores'])
    }



def get_pred(path, infer, cats, valid_tfms=valid_tfms, thresh=0.4):
    pred_dict = arch.end2end_detect(
        load_image(path),
        valid_tfms, 
        infer,
        class_map=ClassMap(cats),
        detection_threshold=thresh)
    pred_dict['path'] = path
    return pred_dict



def get_preds(paths, infer, cats, **kwargs):
    preds = L()
    for i,d in enumerate(paths):
        if i%100==0: print(f"Building preds {i} to {i+100} ...")
        try:
            pred = get_pred(d, infer, cats, **kwargs)
            pred['path'] = d
            preds.append(pred)
        except:
            print(F"Failed to make pred for path[{i}]: {d}")
            pred = _empty_pred()
            pred['path'] = d
            preds.append(pred)
    print(f"Finished building {len(preds)} preds.")
    scores = preds.map(lambda x: max(list(x['detection']['scores']) + [0]))
    plt.hist(scores, 20);
    plt.title("Predictions by Confidence")
    plt.xlabel("Confidence Score")
    plt.ylabel("# of Predictions")
    return preds.map(lambda p: pred_to_rec(p))



def _empty_pred():
    return {'detection':
               {'bboxes': [],
                'scores': [],
                'labels': [],
                'label_ids': []
               },
            'img': None,
            'width': None,
            'height': None}



def get_ap_scores(preds, targs, cats, iou_thresh=0.5):
    """Computes precision and recall for each category using preds and targs.
       Results are returned as a dict with a key for each cat.
    """
    res = {cat: {'ap':0, 'prec':[], 'rec':[], 'n':0} for cat in cats}
    for cat in cats:
        nobjs = 0
        confs, trues = [], []

        for pred, targ in zip(preds, targs):
            conf = pred['confs']
            tbbs = to_xyxy(targ['bboxes'])
            pbbs = to_xyxy(pred['bboxes'])
            true = [0 for bb in pbbs]

            for tbb in tbbs:
                for i,pbb in enumerate(pbbs):
                    if iou(tbb, pbb) >= iou_thresh:
                        true[i] = 1
                        break

            trues = trues + true
            confs = confs + conf
            nobjs += len(tbbs)
            
        _sorted = L(zip(confs, trues)).sorted(lambda x: -x[0])
        confs, trues = _sorted.map(lambda x: x[0]), _sorted.map(lambda x: x[1])
        prec = [sum(trues[:i+1])/(i+1) for i,x in enumerate(trues)]
        rec = [sum(trues[:i+1]) / nobjs for i,x in enumerate(trues)]
        ap = round(trapz(prec, rec), 2)
        res[cat]['prec'], res[cat]['rec'] = prec, rec
        res[cat]['ap'], res[cat]['n'] = ap, nobjs
    return res



def score_model(weights:Path, cats:list):
    """Returns targs, preds, and pr curves given the path to a model & cats.
    """
    print(f"----- Scoring model {weights.name} -----")
    targs = get_targs(cats)
    paths = [t['path'] for t in targs]
    infer = get_infer(weights, cats)
    preds = get_preds(paths, infer, cats, thresh=.10)
    cat2score = get_ap_scores(preds, targs, cats)
    print(f"\n----- Results -----")
    aps = [v['ap'] for v in cat2score.values()]
    mean_ap = sum(aps) / len(aps)
    print(f"mAP: {mean_ap}")
    print(f"By category:")
    for k,v in cat2score.items():
        print(f" • {k} (n={v['n']}): {v['ap']}")
    return targs, preds, cat2score



from numpy import trapz
def plot_prcurve(cat, ap, prec, rec, n):
    fig, ax = plt.subplots()
    ax.plot(rec+[rec[-1]], prec+[0]) # add pt so line ends intersecting x axis
    ax.set(title=f'{cat}  |  AP={ap}  |  # target objs={n}',
           xlabel='Recall',
           ylabel='Precision')
    plt.axis([0, 1.05, 0, 1.05]);
    

    
    
    
    
    
    
# -----------------------------------------------------------------------------
# -----
# -----     Datasets & Dataloaders
# -----


class MyParser(Parser):
    
    def __init__(self, recs, class_list):
        super().__init__(template_record = ObjectDetectionRecord())
        self.recs = recs
        self.class_map = ClassMap(class_list)
        
    def __iter__(self) -> Any:
        for rec in iter(self.recs):
            yield rec
    
    def __len__(self) -> int:
        return len(self.recs)
    
    def record_id(self, rec: Any) -> Hashable:
        return rec['path']
    
    def parse_fields(self, o: Any, record: BaseRecord, is_new: bool):
        record.set_img_size(ImgSize(height=o['height'], width=o['width']))
        record.set_filepath(o['path'])
        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(o['cats'])
        record.detection.add_bboxes([BBox.from_xywh(*bbox) for bbox in o['bboxes']])

        

def get_dataloaders(dss, model, show_batch=False):
    t_ds, v_ds = dss
    t_dl = arch.train_dl(t_ds, batch_size=bs, num_workers=4, shuffle=True)
    v_dl = arch.valid_dl(v_ds, batch_size=bs, num_workers=4, shuffle=False)
    if show_batch:
        print("Showing first batch from valid_dl ...")
        arch.show_batch(first(v_dl), ncols=4)
    return [t_dl, v_dl]



def get_datasets(subset, cats, train_tfms=train_tfms, valid_tfms=valid_tfms):
    parser = MyParser(subset, cats)
    t_recs, v_recs = parser.parse()
    t_ds = Dataset(t_recs, train_tfms)
    v_ds = Dataset(v_recs, valid_tfms)
    return [t_ds, v_ds]







# -----------------------------------------------------------------------------
# -----
# -----     COCO Data Utils
# -----


def get_coco_annos(path, cats=None, prefix=None):
    """
    Open a COCO style json in `path` and returns the lists of filenames (with
    maybe `prefix`) and labeled bboxes.
    """
    coco = load_json(path)
    id2img, id2bbs, id2lbls = {}, defaultdict(list), defaultdict(list)
    id2cat = {o['id']:o['name'] for o in coco['categories']}
    
    # populate id2bbs and id2lbls
    for o in coco['annotations']:
        bb, lbl = bb_to_xyxy(o['bbox']), id2cat[o['category_id']]
        if cats == None:
            id2bbs[o['image_id']].append(bb)
            id2lbls[o['image_id']].append(lbl)
        else:
            if lbl in cats:
                id2bbs[o['image_id']].append(bb)
                id2lbls[o['image_id']].append(lbl)
    
    # populate id2img
    if 'file_name' in coco['images'][0].keys(): # LVIS doesn't have 'file_name'
        id2img = {o['id'] : ifnone(prefix,'') + o['file_name']
                  for o in coco['images'] if o['id'] in id2lbls}
    else:
        id2img = {o['id'] : ifnone(prefix,'') + Path(o['coco_url']).name
                  for o in coco['images'] if o['id'] in id2lbls}
    
    # iterate through IDs to return img and annos
    ids = list(id2img.keys())
    fns = [id2img[i] for i in ids] 
    annos = [(id2bbs[i], id2lbls[i]) for i in ids]
    return fns, annos



def get_nacti_annos(path, cats=None, prefix=None):
    """
    Open a NACTI json in `path` and returns the lists of filenames (with
    maybe `prefix`). Unlike COCO, NACTI labels are single-label.
    Returns fns, lbls.
    """
    cct = load_json(path)
    id2img, id2lbl = {}, defaultdict(None)
    id2cat = {o['id']:o['common name'] for o in cct['categories']}
    
    # populate id2lbl
    for o in cct['annotations']:
        lbl = id2cat[o['category_id']]
        if cats == None:
            id2lbl[o['image_id']] = lbl
        else:
            if lbl in cats:
                id2lbl[o['image_id']] = lbl
    
    # populate id2img
    id2img = {o['id'] : ifnone(prefix,'') + o['file_name']
              for o in cct['images'] if o['id'] in id2lbl}
    
    # iterate through IDs to return img and annos
    ids = list(id2img.keys())
    return [id2img[i] for i in ids], [id2lbl[i] for i in ids]



def get_coco_subset(cat_map, n=None, **kwargs):
    """
    cat_map is a map from one of my cats to a list of coco cats.
    Ex: {'animal': ['cow','sheep','deer'] , 'person': ['person']}
    """
    cats = []
    for v in cat_map.values():
        cats = cats + v
        
    flat = json.load(open(COCO_FLAT))
    filtered = filter_by_cats(flat, cats)
    filtered = filter_by_max_objs(filtered)
    remapped = remap_annos(filtered, cat_map, **kwargs)
    
    res = L(remapped.values())
    res = res if n == None else res.shuffle()[:n]
    
    print(f"Finished getting COCO subset:")
    print("\tn:\t\t", len(res))
    print("\told cats:", cats)
    print("\tnew cats:", list(cat_map.keys()))
    return res



def filter_by_cats(flat, cats, verbose=False):
    if verbose: print(f"Filtering records by categories {cats} ...")
        
    n_in = len(flat)
    # loop through records to remove annos that aren't in cats
    for img_id, rec in flat.items():
        
        # accumulators to store filtered cats & bboxes
        new_cats   = []
        new_bboxes = []
        
        # we'll loop through the old cats to do the filtering
        old_cats   = rec['cats']
        old_bboxes = rec['bboxes']

        for idx, cat in enumerate(old_cats):
            if cat in cats:   
                new_cats.append(cat)
                new_bboxes.append(old_bboxes[idx])
                
        flat[img_id]['cats']   = new_cats
        flat[img_id]['bboxes'] = new_bboxes
        
    # remove records with no cats left
    res = L(flat.items()).filter(lambda i: len(i[1]['cats']) > 0)
    res = {img:rec for (img,rec) in res}
    if verbose: print(f"Finished category filter:\n\tn:\t{n_in}\t→ {len(res)}")
    return res



def filter_by_max_objs(flat, max_objs=10, verbose=False):
    if verbose: print(f"Filtering records by max_objs {max_objs} ...")
        
    # get starting info
    img2ct = {img: len(rec['bboxes']) for img,rec in flat.items()}
    cts = L(img2ct.values())
    s = [len(flat), max(cts), round(sum(cts)/len(cts),1), min(cts)]
    
    # apply filter
    res = L(flat.items()).filter(lambda i: img2ct[i[0]] <= max_objs)
    res = {k:v for k,v in res}
    
    # get ending info
    img2ct = {img: len(rec['bboxes']) for img,rec in res.items()}
    cts = L(img2ct.values())
    e = [len(res), max(cts), round(sum(cts)/len(cts),1), min(cts)]
    if verbose:
        print("Finished max_objs filter:")
        print(f"\tn:\t{s[0]}\t→ {e[0]}\n\tmax:\t{s[1]}\t→ {e[1]}\n\tavg:\t{s[2]}\t→ {e[2]}")
    return res



def remap_annos(annos, cat_map):
    """
    cat_map is a map from one of my cats to a list of coco cats.
    Ex: {'animal': ['cow','sheep','deer'] , 'person': ['person']}
    """
    # get map from new category to id
    cat2id = {k:i for i,k in enumerate(cat_map.keys())}
    
    # reverse the k,v's and flatten (e.g. each animal has a separate key)
    inv_map = {}
    for new, coco_cats in cat_map.items():
        for coco_cat in coco_cats:
            inv_map[coco_cat] = new

    # apply the two maps created above
    for img_id, rec in annos.items():
        rec['cats'] = L(rec['cats']).map(lambda o: inv_map[o])
        rec['cat_ids'] = L(rec['cats']).map(lambda o: cat2id[o])
    
    return annos






# -----------------------------------------------------------------------------
# -----
# -----     Displaying Images
# -----


def display_images(paths, cols=3, figsize=(15,15), max_n=12):
    """
    Display images from a list of paths.
    """
    n_imgs = len(paths)
    n      = min(n_imgs, max_n)
    cols   = min(cols, n)
    
    print(f"Displaying {n} of {n_imgs} images.")
    
    if n == 0:
        None
    elif n == 1:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(figsize)
        ax.axis('off')
        ax.imshow(load_image(paths[0]))
        plt.tight_layout()
    elif 1 < n <= cols:
        fig, axs = plt.subplots(1, cols)
        fig.set_size_inches(figsize)
        for i in range(n):
            axs[i].axis('off')
            axs[i].imshow(load_image(paths[i]))
        plt.tight_layout()
    else:  
        extra_row = 1 if n%cols > 0 else 0
        fig, axs = plt.subplots(n//cols + extra_row, cols)
        fig.set_size_inches(figsize)
        for i in range(n):
            row, col = i//cols, i%cols
            axs[row][col].axis('off')
            axs[row][col].imshow(load_image(paths[i]))
        plt.tight_layout()
        
        
        
def display_preds(preds, cols=2, figsize=(20,20), max_n=10):
    """preds is a list of dicts with key 'img'
    """
    n = len(preds)
    if n > max_n:
        print(f"Showing first {max_n} images")
        n = max_n
        
    extra_row = 1 if n % cols > 0 else 0
    fig, axs = plt.subplots(n//cols + extra_row, cols)
    fig.set_size_inches(figsize)
    
    for i in range(n):
        row, col = i//cols, i%cols
        axs[row][col].axis('off')
        axs[row][col].imshow(preds[i]['img'])
        
    plt.tight_layout()

        

def display_rec(rec, figsize=(15,15)):
    """
    rec:dict, must contain keys 'path', 'bboxes', and 'cats'.
    display_rec uses fast.ai's built-in show method for LabeledBBox.
    """ 
    lbl, bb = rec['cats'], rec['bboxes']
    tbb = TensorBBox(to_xyxy(bb))
    lblbb = LabeledBBox(tbb, L(lbl))
    
    im = PILImage.create((rec['path']))
    ctx = im.show(figsize=figsize, cmap='Greys')
    lblbb.show(ctx=ctx);

    
    
    
    
    
    
# -----------------------------------------------------------------------------
# -----
# -----     Utils
# -----
    
    
def bb_to_xyxy(bb):
    return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
    
def to_xyxy(bbs):
    return [bb_to_xyxy(b) for b in bbs]


def bb_to_xywh(bb):
    """Single bbox from xyxy to xywh"""
    return [bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]]

def to_xywh(bbs):
    return [bb_to_xywh(b) for b in bbs]



a = [0,0,10,10]
b = [2,2,4,4]
c = [0,0,-10,-10]
d = [-5,1,0,2]



@typed
def icebb_to_list(bb:list) -> list:
    return [[b.xmin, b.ymin, b.xmax, b.ymax] for b in bb]
test_eq(icebb_to_list([BBox(*a),BBox(*b)]) , [a,b])



@typed
def validate_line_segment(l) -> None:
    start, end = l
    if start > end:
        raise Exception(f"(min,max) tuple {l} has min > max but needs min < max.")
    elif start == end:
        raise Exception(f"(min,max) tuple {l} has min = max but needs min < max.")
validate_line_segment((1,2))
validate_line_segment((-1,1))
with exception: validate_line_segment((10,9))
with exception: validate_line_segment((0,0))


    
@typed
def line_segment_overlap(a:tuple, b:tuple) -> float:
    validate_line_segment(a), validate_line_segment(b)
    a_starts_in = b[0] <= a[0] <= b[1]
    a_ends_in   = b[0] <= a[1] <= b[1]
    if       a_starts_in and     a_ends_in: res = a[1] - a[0] # a in b
    elif     a_starts_in and not a_ends_in: res = b[1] - a[0] # some overlap
    elif not a_starts_in and     a_ends_in: res = a[1] - b[0] # some overlap
    else:
        b_starts_in = a[0] <= b[0] <= a[1]
        if b_starts_in: res = b[1] - b[0] # b in a
        else:           res = 0           # no overlap
    return float(max(0, res))
test_eq( line_segment_overlap((-10,10),(0,3)) , 3. )
test_eq( line_segment_overlap((0,4),(2,5)), 2. )
test_eq( line_segment_overlap((1,2),(2,3)), 0. )
with exception: line_segment_overlap((0,10),(2,1))
with exception: line_segment_overlap((0,10),(0,0))
    
    
    
@typed
def validate_bbox(b) -> None:
    validate_line_segment((b[0], b[2]))
    validate_line_segment((b[1], b[3]))
validate_bbox(a)
validate_bbox(b)
with exception: validate_bbox(c)
validate_bbox(d)



@typed
def bbox_intersection(a, b) -> float:
    # a and b are bboxes in xyxy form.
    ax , ay = (a[0], a[2]) , (a[1], a[3])
    bx , by = (b[0], b[2]) , (b[1], b[3])
    x_overlap = line_segment_overlap(ax, bx)
    y_overlap = line_segment_overlap(ay, by)
    return x_overlap * y_overlap
test_eq( bbox_intersection(a,b) , 4. )
with exception: bbox_intersection(a,c)
test_eq( bbox_intersection(a,d) , 0. )



@typed
def bbox_area(b:list) -> float:
    validate_bbox(b)
    return float((b[2]-b[0])*(b[3]-b[1]))
test_eq(bbox_area(a), 100.)
test_eq(bbox_area(b), 4.)
with exception: bbox_area(c)
test_eq(bbox_area(d), 5.)



@typed
def iou(a:list, b:list) -> float:
    validate_bbox(a)
    validate_bbox(b)
    intersection = bbox_intersection(a,b)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union
test_eq(iou([0,0,10,10], [0,0,5,5]) , .25)
test_eq(iou([0,0,1,1], [5,5,6,6]) , 0)
with exception: iou([0,0,10,-10], [0,0,5,5])
with exception: iou([0,0,-10,-10], [0,0,5,5])

    

def print_mem():
    total      = torch.cuda.get_device_properties(0).total_memory
    reserved   = torch.cuda.memory_reserved(0)
    allocated  = torch.cuda.memory_allocated(0)
    in_use  = allocated + reserved
    free    = total - in_use
    gig = 1_073_741_824
    print(f"{round(in_use/gig,2)} GB USED")
    print(f"{round(free/gig,2)} GB FREE")
    
    
    
@contextmanager
def be_quiet():
    """
    Ex:
      with be_quiet(): print("You won't see this ...")
      print("... But you will see this!")
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



def get_date():
    now = datetime.now()
    return now.strftime("%Y-%m-%d")





# -----------------------------------------------------------------------------
# -----
# -----     Saving & Loading
# -----
    

def save_txt(iterable, path, sep="\n", add_date=False):
    if add_date: path = append_date(path)
    with open(path, "w") as f:
        for obj in iterable:
            f.write(obj + sep)
    print(f"Saved to: {path}")

def load_txt(path, sep="\n"):
    file = open(path, "r")
    res = file.read().split(sep)
    file.close()
    if res[-1] == '':
        return res[:-1]
    else:
        return res

    

def save_json(obj, path, add_date=False):
    if add_date: path = append_date(path)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"Saved to: {path}")
        
def load_json(path):
    return json.load(open(path))


             
def save_pickle(obj, path, add_date=False):
    if add_date: path = append_date(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to: {path}")
        
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

    
       
def save_weights(model, path, add_date=True):
    if add_date: path = append_date(path)
    torch.save(model.state_dict(), path)
    print(f"Saved to: {path}")
    
    
    
def load_csv(path):
    return pd.read_csv(path)


