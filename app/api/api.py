import pandas as pd
from urllib.parse import urlencode
from fastapi import APIRouter, Form, Request, HTTPException
from fastapi.responses import ORJSONResponse, RedirectResponse
from app.core.registry import get_store
from app.core.templates import templates
from app.utils.input_builder import build_base
from app.utils.ai_analyzer import PropertyInvestmentAnalyzer
from app.schemas.pandera_ import StructuralSchema
from app.model.helpers import coerce_df_to_match_schema
from app.utils.helpers import extract_address_from_url
from app.utils.perm_builder import (
    assemble_options,
    get_original_match_mask,
    scenario_df_verify,
    SCENARIO_DEFAULTS
)

router = APIRouter(prefix="/api")


@router.post("/from_url")
async def from_url(url: str = Form(...)):
    # minimal: just pass original URL to output and let it compute there
    qs = urlencode({"url": url})
    return RedirectResponse(url=f"/output?{qs}", status_code=303)


@router.post("/preds", response_class=ORJSONResponse)
async def preds_from_url(
    url: str = Form(..., description="Zillow or Redfin listing URL"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: float = Form(..., description="Number of bathrooms"),
    accommodates: int = Form(..., description="Max guest capacity"),
):
    """
    Build a base listing from a Zillow/Redfin URL and user-provided overrides,
    only return predictions.
    """
    store = get_store()

    address = extract_address_from_url(url)

    base = build_base(address=address)
    base.update(SCENARIO_DEFAULTS)

    input_values = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "beds": bedrooms,
    }

    base.update(input_values)
    base_df = scenario_df_verify(pd.DataFrame(base, index=[0]))
    base_df["dist_to_bus_km"] = base_df["dist_to_bus_km"].fillna(50)
    base_df, changes = coerce_df_to_match_schema(base_df, StructuralSchema)

    preds = store.pipeline.predict(base_df).to_dict()

    return ORJSONResponse(content=preds, status_code=200)


@router.post("/perms")
async def perms_from_url(
    request: Request,
    url: str = Form(..., description="Zillow or Redfin listing URL"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: float = Form(..., description="Number of bathrooms"),
    accommodates: int = Form(..., description="Max guest capacity"),
):
    """
    Build a base listing from a Zillow/Redfin URL and user-provided overrides,
    generate permutations, run prediction + uplift, and return a
    frontend-friendly payload.
    """
    try:
        store = get_store()

        # 1) Get address (and possibly other metadata) from the URL
        address = extract_address_from_url(url)

        # 2) Build the base row, injecting user overrides
        # Adjust this to match your actual build_base signature
        base = build_base(address=address)
        base.update(SCENARIO_DEFAULTS)
        base["dist_to_bus_km"] = 50

        input_values = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "accommodates": accommodates,
            "beds": bedrooms,
        }

        # 3) Generate permutations + run predictions + explanations
        permo_df = assemble_options(base, input_values)
        permo_df["dist_to_bus_km"] = permo_df["dist_to_bus_km"].fillna(50)
        permo_df = scenario_df_verify(permo_df)
        permo_df, _ = coerce_df_to_match_schema(
            permo_df, StructuralSchema)
        preds_and_exp = store.pipeline.predict_and_explain(permo_df)

        # 4) Shape a clean response for the UI
        idx = get_original_match_mask(permo_df, input_values).index[0]
        content = {
            "request": request,
            "address": str(address),
            "price_pred": preds_and_exp["preds"].at[idx, "price_pred"],
            "occ_pred": int(preds_and_exp["preds"].at[idx, "occ_pred"]),
            "revenue": preds_and_exp["preds"].at[idx, "rev_final_pred"],
            "uplift_table": preds_and_exp["uplift_table"].to_dict(),
            "uplift_chart_png": preds_and_exp["uplift_char_png"],
        }

        return templates.TemplateResponse(
            "index.html", content,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process URL: {e}")


@router.post("/preds_w_AI")
async def preds_with_ai(
    request: Request,
    url: str = Form(..., description="Zillow or Redfin listing URL"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: float = Form(..., description="Number of bathrooms"),
    accommodates: int = Form(..., description="Max guest capacity"),
):
    if not url:
        raise HTTPException(400, "Missing url")
    try:

        store = get_store()
        print("*** ASSEMBLING INPUT")
        address = extract_address_from_url(url)
        base = build_base(address=address)
        base.update(SCENARIO_DEFAULTS)

        input_values = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "accommodates": accommodates,
            "beds": bedrooms,
        }

        base.update(input_values)
        base_df = scenario_df_verify(pd.DataFrame(base, index=[0]))
        base_df["dist_to_bus_km"] = base_df["dist_to_bus_km"].fillna(50)
        base_df, _ = coerce_df_to_match_schema(base_df, StructuralSchema)
        prop_dict = base_df.iloc[0].to_dict()

        print("*** MAKING PREDICTIONS")
        pred_df = store.pipeline.predict(base_df)
        preds = pred_df.iloc[0].to_dict()

        lat = address.latitude
        lon = address.longitude

        price_pred_raw = preds.get('price_pred', 0)
        occ_pred_raw = preds.get("occ_pred", 0)
        rev_pred_raw = preds.get("rev_final_pred", 0)

        maps_url = (
            f"https://www.google.com/maps?q={lat},{lon}"
            if lat is not None and lon is not None
            else f"https://www.google.com/maps/search/{quote(address)}"
        )

        # Generate AI investment summary
        ai_summary = None
        ai_error = None
        print("*** GETTING AI SUMMARY")
        try:
            analyzer = PropertyInvestmentAnalyzer()

            ai_summary = analyzer.generate_investment_summary(
                address=address.address,
                price_pred=price_pred_raw,
                occ_pred=occ_pred_raw,
                rev_pred=rev_pred_raw,
                property_features=prop_dict
            )
        except Exception as e:
            # If AI fails, continue without it (ML predictions still show)
            ai_error = f"AI analysis unavailable: {str(e)}"

        content = {
            "request": request,
            "prop_dict": prop_dict,
            "price_pred": price_pred_raw,
            "occ_pred": occ_pred_raw,
            "rev_pred": rev_pred_raw,
            "address": address,
            "maps_url": maps_url,
            "ai_summary": ai_summary,
            "ai_error": ai_error,
        }

        return templates.TemplateResponse(
            "ai_index.html", content
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process URL: {e}")
