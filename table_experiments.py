import math
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

COLORS = {
    "text": "#0d0d42",
    "label": "#2d3579",
    "subheader": "#3e49a7",
    "primary": "#00a9de",
    "selectedRow": "#b5d0ff",
    "subheaderBackground": "#95a0ff",
    "evenRow": "#eaecff",
    "primaryBackground": "#eaecff",
    "secondaryBackground": "#f0f2f6",
    "background": "#ffffff",
    "red": "#ff2255",
}


def main():
    def color_negative_red(val):
        color = "red" if val < 0 else "black"
        return "color: %s" % color

    def highlight_max(data, color="yellow"):
        """highlight the maximum in a Series or DataFrame"""
        attr = "background-color: {}".format(color)
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = data == data.max()
            return [attr if v else "" for v in is_max]
        else:  # from .apply(axis=None)
            is_max = data == data.max().max()
            return pd.DataFrame(np.where(is_max, attr, ""), index=data.index, columns=data.columns)

    def color_selected_row(row: pd.Series):
        if row.name == st.session_state.get("selected_row", 0):
            return [f"background-color: {COLORS['selectedRow']}" for v in row]
        return [f"" for v in row]

    np.random.seed(24)
    df = pd.DataFrame({"A": np.linspace(1, 20, 20)})
    df = pd.concat([df, pd.DataFrame(np.random.randn(20, 4), columns=list("BCDE"))], axis=1)
    df.iloc[0, 2] = np.nan
    df = df.round(4)

    st.title("st.dataframe")
    st.markdown(
        """
        Interactions: Not in the table directly but in components around :warning: (planned for [Aug - Oct 2023](https://roadmap.streamlit.app/#aug-oct-2023))  
        Styles: Available on cells :white_check_mark:  
        Limitation: Columns header cannot be styled (in development with [Column configs](https://roadmap.streamlit.app))
        """
    )
    st.dataframe(
        df.style.format("{:.2%}")
        .set_properties(**{"background-color": COLORS["primaryBackground"], "color": COLORS["text"]})
        .applymap(color_negative_red)
        .apply(color_selected_row, axis=1)
        .apply(highlight_max, color="darkorange", axis=0),
        use_container_width=True,
        height=420,
    )
    cols = st.columns([2, 10])
    cols[0].number_input(
        "Selected row",
        value=0,
        min_value=df.index.min(),
        max_value=df.index.max(),
        key="selected_row",
    )

    st.title("st.experiment_data_editor")
    st.markdown(
        """
        Interactions: yes :white_check_mark:  
        Styles: None :x:  
        Limitation: All columns are editable (in development with [Column configs](https://roadmap.streamlit.app))
        """
    )

    if "source_df" not in st.session_state:
        st.session_state["source_df"] = df.assign(selected=False)[["selected"] + list("ABCD")]

    result = st.experimental_data_editor(
        st.session_state["source_df"],
        key="df_state",
        # on_change=st.experimental_rerun,
        height=420,
        use_container_width=True,
        # num_rows="dynamic",
    )

    st.markdown("Selected rows:")
    st.dataframe(
        result[result["selected"] == True][list("ABCD")],
        # height=420,
        use_container_width=True,
    )

    st.title("st.columns")
    st.markdown(
        """
        Interactions: yes with buttons or other elements :white_check_mark:  
        Styles: Limited to text color :x:  
        Limitations: Not actually a table, no scroll
        """
    )
    if "clicked_row" not in st.session_state:
        st.session_state["clicked_row"] = None

    def select_row(idx):
        return lambda: st.session_state.update(clicked_row=idx)

    limit, offset = pagination_input(total_count=len(df), page_size=7)
    text_table(
        df.iloc[limit : limit + offset],
        key="text_table",
        action_column=lambda col, idx, row: col.button("Show", key=f"edit-{idx}", on_click=select_row(idx + 1)),
    )
    if st.session_state["clicked_row"]:
        st.markdown(f"**You clicked on {st.session_state['clicked_row']}**")

    st.title("st.plotly_chart table")
    st.markdown(
        """
        Interactions: None :x:  
        Styles: Yes :white_check_mark:  
        """
    )

    # https://plotly.com/python/table/#styled-table

    import plotly.graph_objects as go

    headerColor = COLORS["primary"]
    rowEvenColor = COLORS["evenRow"]
    rowOddColor = "white"

    df.to_dict("list").values()
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    line_color="darkslategray",
                    fill_color=headerColor,
                    align=["left", "center"],
                    font=dict(color="white", size=18),
                    height=32,
                ),
                cells=dict(
                    values=list(df.to_dict("list").values()),
                    line_color="darkslategray",
                    fill_color=[[rowOddColor, rowEvenColor] * 10],
                    align=["left", "center"],
                    font=dict(color="darkslategray", size=14),
                    height=25,
                ),
            )
        ]
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, height=420)


def color_negative(val):
    return f":red[{val}]" if val < 0 else val


def text_table(dataframe: pd.DataFrame, key: str, action_column: Callable[[DeltaGenerator, int, Dict[str, Any]], None]):
    if key not in st.session_state:
        st.session_state[key] = None

    nb_columns = len(dataframe.columns) + 1

    header_cols = st.columns(nb_columns)
    for col_index, column_name in enumerate(list(dataframe.columns) + ["Action"]):
        header_cols[col_index].markdown(f"**{column_name}**")

    for row_index, row in dataframe.iterrows():
        row_object = row.to_dict()
        row_cols = st.columns([1] * len(row_object) + [1])

        for col_index, value in enumerate(row_object.values()):
            row_cols[col_index].markdown(color_negative(value))

        action_column(row_cols[-1], row_index, row_object)

    return st.session_state[key]


def pagination_input(total_count: int, page_size: int = 5, key="page") -> Tuple[int, int]:
    nb_pages = math.ceil(total_count / page_size)

    cols = st.columns([2, 1, 1] + [1] * 8)
    page = cols[0].number_input("Page", value=1, key=key, min_value=1, max_value=nb_pages)
    cols[1].markdown("#")
    cols[1].markdown(f"**/ {nb_pages}**")

    # cols[2].markdown("##")
    # cols[2].button("⬅️", on_click=lambda: st.session_state.update(**{key: max(page - 1, 1)}), use_container_width=True)

    # cols[3].markdown("##")
    # cols[3].button(
    #     "➡️", on_click=lambda: st.session_state.update(**{key: min(page + 1, nb_pages)}), use_container_width=True
    # )

    offset = (page - 1) * page_size
    limit = page_size
    return offset, limit


if __name__ == "__main__":
    main()
