from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import point
import filter
import segment
import restoration
import fancy
from pydantic import BaseModel
from typing import Optional
from utils import str_id, export_image
app = FastAPI()

origins = ["*"]
exports_folder = "exports/*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Image(BaseModel):
    uri: str
    name: str
    x: Optional[int] = None
    k: Optional[int] = None
    a: Optional[int] = None
    b: Optional[int] = None
    c: Optional[int] = None


@app.post("/point/reverse")
async def reverse(image: Image):
    img = point.reverse_image(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/point/threshold")
async def threshold(image: Image):
    img = point.threshold(image.uri, image.a, image.b)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/point/log")
async def log_transformation(image: Image):
    img = point.log_transformation(image.uri, image.c)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/point/hist")
async def hist(image: Image):
    img = point.hist(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/filter/gaussian-blur")
async def gaussian_blur(image: Image):
    img = filter.gaussian_blur(image.uri, image.x)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/filter/laplacian")
async def laplacian(image: Image):
    img = filter.laplacian(image.uri, image.k)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/filter/sobel")
async def sobel(image: Image):
    img = filter.sobel(image.uri, image.k)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/segment/kmean")
async def kmean(image: Image):
    img = segment.kmean(image.uri, image.k)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


# @app.post("/segment/graph-cut")
# async def graph_cut(image: Image):
#     img = segment.graph_cut(image.uri)
#     name = str_id()+image.name
#     export_image(img, "exports/"+name)
#     return {"filename": name}


# @app.post("/segment/meanshift")
# async def meanshift(image: Image):
#     img = segment.mean_shift(image.uri)
#     name = str_id()+image.name
#     export_image(img, "exports/"+name)
#     return {"filename": name}


@app.post("/restoration/tb-so-hoc")
async def tb_so_hoc(image: Image):
    img = restoration.tb_so_hoc(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/tb-hinh-hoc")
async def tb_hinh_hoc(image: Image):
    img = restoration.tb_hinh_hoc(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/tb-harmonic")
async def tb_harmonic(image: Image):
    img = restoration.tb_harmonic(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/tb-contraharmonic")
async def tb_contraharmonic(image: Image):
    img = restoration.tb_contraharmonic(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-trung-vi")
async def loc_trung_vi(image: Image):
    img = restoration.loc_trung_vi(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-min")
async def loc_min(image: Image):
    img = restoration.loc_min(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-max")
async def loc_max(image: Image):
    img = restoration.loc_max(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-midpoint")
async def loc_midpoint(image: Image):
    img = restoration.loc_midpoint(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-alpha")
async def loc_alpha(image: Image):
    img = restoration.loc_alpha(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-tuong-thich")
async def loc_tuong_thich(image: Image):
    img = restoration.loc_tuong_thich(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/restoration/loc-thong-thap")
async def loc_tuong_thich(image: Image):
    img = restoration.loc_thong_thap_ly_tuong(image.uri, 20000)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/fancy/warming")
async def Warming(image: Image):
    img = fancy._warming(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/fancy/cooling")
async def Cooling(image: Image):
    img = fancy._cooling(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/fancy/moon")
async def Moon(image: Image):
    img = fancy._moon(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/fancy/cartoon")
async def Cartoon(image: Image):
    img = fancy._cartoon(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.post("/fancy/pencil-sketch-grey")
async def Sketch_pencil_using_blending(image: Image):
    img = fancy._sketch_pencil_using_blending(image.uri)
    name = str_id()+image.name
    export_image(img, "exports/"+name)
    return {"filename": name}


@app.get("/exports/{filename}")
def get_file(filename: str):
    return FileResponse("exports/"+filename)
