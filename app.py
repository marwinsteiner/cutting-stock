import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import uuid


class WoodStock:
    def __init__(self):
        if 'stock' not in st.session_state:
            # Initialize with proper dtypes
            st.session_state.stock = pd.DataFrame({
                'id': pd.Series(dtype='str'),
                'length': pd.Series(dtype='float64'),
                'remaining_length': pd.Series(dtype='float64'),
                'width': pd.Series(dtype='float64'),
                'status': pd.Series(dtype='str')
            })
        if 'cut_history' not in st.session_state:
            st.session_state.cut_history = []


    def add_stock(self, length: float, width: float, quantity: int):
        # Create new stock with explicit dtypes
        new_stock_data = {
            'id': [str(uuid.uuid4()) for _ in range(quantity)],
            'length': [length] * quantity,
            'remaining_length': [length] * quantity,
            'width': [width] * quantity,
            'status': ['uncut'] * quantity
        }
        new_stock = pd.DataFrame(new_stock_data)

        # Concatenate and update session state
        if st.session_state.stock.empty:
            st.session_state.stock = new_stock
        else:
            st.session_state.stock = pd.concat(
                [st.session_state.stock, new_stock],
                ignore_index=True,
                verify_integrity=True
            )

    def get_available_stock(self):
        # Return a formatted version of the stock for display
        stock_df = st.session_state.stock.copy()

        # Create a formatted display DataFrame
        display_df = stock_df[[
            'length', 'remaining_length', 'width', 'status'
        ]].copy()

        # Rename columns for better display
        display_df.columns = ['Original Length', 'Remaining Length', 'Width', 'Status']

        # Sort by status (uncut first, then partially used)
        display_df = display_df.sort_values(
            by=['Status', 'Remaining Length'],
            ascending=[True, False]
        )

        # Format numbers to 2 decimal places
        display_df['Original Length'] = display_df['Original Length'].round(2)
        display_df['Remaining Length'] = display_df['Remaining Length'].round(2)
        display_df['Width'] = display_df['Width'].round(2)

        # Add units to dimensions
        display_df['Original Length'] = display_df['Original Length'].astype(str) + '"'
        display_df['Remaining Length'] = display_df['Remaining Length'].astype(str) + '"'
        display_df['Width'] = display_df['Width'].astype(str) + '"'

        # Capitalize status
        display_df['Status'] = display_df['Status'].str.replace('_', ' ').str.title()

        return display_df

    def get_stock_for_processing(self):
        # Return the original DataFrame for calculations
        return st.session_state.stock[st.session_state.stock['remaining_length'] > 0]

    def update_stock(self, cuts: List[Tuple[str, List[Dict[str, float]]]]):
        # Create a copy of the current stock
        updated_stock = st.session_state.stock.copy()

        for stock_id, cut_pieces in cuts:
            stock_piece = updated_stock[updated_stock['id'] == stock_id].iloc[0]
            total_length_used = sum(piece['length'] for piece in cut_pieces)
            remaining = stock_piece['remaining_length'] - total_length_used

            # Update the copy
            updated_stock.loc[updated_stock['id'] == stock_id, 'remaining_length'] = remaining
            updated_stock.loc[updated_stock['id'] == stock_id, 'status'] = 'partially used' if remaining > 0 else 'used'

        # Update session state with the new DataFrame
        st.session_state.stock = updated_stock


def check_stock_availability(required_pieces: List[Dict[str, float]], available_stock: pd.DataFrame) -> Tuple[
    bool, List[Dict[str, float]]]:
    missing_pieces = []

    for piece in required_pieces:
        suitable_stock = available_stock[
            (available_stock['remaining_length'] >= piece['length']) &
            (available_stock['width'] >= piece['width'])
            ]

        if suitable_stock.empty:
            missing_pieces.append(piece)

    return len(missing_pieces) == 0, missing_pieces


def solve_cutting_stock(required_pieces: List[Dict[str, float]], available_stock: pd.DataFrame) -> Tuple[
    List[Tuple[str, List[Dict[str, float]]]], List[Dict[str, float]]]:
    solution = []
    remaining_pieces = required_pieces.copy()
    used_stock_ids = set()

    # Sort pieces in descending order by length and width
    remaining_pieces.sort(key=lambda x: (x['length'], x['width']), reverse=True)

    while remaining_pieces:
        piece_fitted = False

        # Filter out already used stock
        available_df = available_stock[~available_stock['id'].isin(used_stock_ids)]

        if available_df.empty:
            break

        # First try to find exact width matches, then wider pieces
        # Sort available stock by width (ascending) to prefer exact matches
        available_df = available_df.sort_values('width', ascending=True)

        for _, stock in available_df.iterrows():
            if stock['remaining_length'] <= 0:
                continue

            current_stock_cuts = []
            remaining_length = stock['remaining_length']
            stock_width = stock['width']

            # Try to fit pieces into current stock
            i = 0
            while i < len(remaining_pieces):
                piece = remaining_pieces[i]

                # Check if piece can fit (both length and width)
                if (piece['length'] <= remaining_length and
                        piece['width'] <= stock_width):

                    current_stock_cuts.append({
                        'length': piece['length'],
                        'width': piece['width'],
                        'width_cut_needed': piece['width'] < stock_width
                    })
                    remaining_length -= piece['length']
                    remaining_pieces.pop(i)
                    piece_fitted = True
                else:
                    i += 1

            if current_stock_cuts:
                solution.append((stock['id'], current_stock_cuts))
                used_stock_ids.add(stock['id'])
                break

        if not piece_fitted:
            break

    return solution, remaining_pieces


def main():
    st.title("Wood Cutting Stock Optimizer")

    # Initialize wood stock manager
    stock_manager = WoodStock()

    # Sidebar for stock management
    st.sidebar.header("Stock Management")

    # Add new stock form
    with st.sidebar.form("add_stock_form"):
        length = st.number_input("Length (inches)", min_value=1.0, value=96.0)
        width = st.number_input("Width (inches)", min_value=1.0, value=4.0)
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

        if st.form_submit_button("Add Stock"):
            stock_manager.add_stock(length, width, quantity)
            st.success("Stock added successfully!")

    # Display current stock with improved formatting
    st.sidebar.subheader("Current Stock")
    current_stock_display = stock_manager.get_available_stock()
    st.sidebar.dataframe(
        current_stock_display,
        hide_index=True,
        use_container_width=True
    )

    # Display stock statistics
    total_pieces = len(st.session_state.stock)
    uncut_pieces = len(st.session_state.stock[st.session_state.stock['status'] == 'uncut'])
    partially_used = len(st.session_state.stock[st.session_state.stock['status'] == 'partially used'])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stock Statistics")
    st.sidebar.markdown(f"- Total pieces: {total_pieces}")
    st.sidebar.markdown(f"- Uncut pieces: {uncut_pieces}")
    st.sidebar.markdown(f"- Partially used: {partially_used}")

    # Main window for cutting optimization
    st.header("Cut Optimizer")

    # Input required pieces
    st.subheader("Enter Required Pieces")
    col1, col2, col3 = st.columns(3)

    with col1:
        new_length = st.number_input("Length needed (inches)", min_value=1.0, value=24.0)
    with col2:
        new_width = st.number_input("Width needed (inches)", min_value=0.5, value=2.0)
    with col3:
        new_quantity = st.number_input("Quantity needed", min_value=1, value=1, step=1)

    if st.button("Add to Requirements"):
        if 'requirements' not in st.session_state:
            st.session_state.requirements = []
        new_pieces = [{'length': new_length, 'width': new_width} for _ in range(new_quantity)]
        st.session_state.requirements.extend(new_pieces)

    # Display current requirements
    if 'requirements' in st.session_state and st.session_state.requirements:
        st.write("Current Requirements:")
        for i, req in enumerate(st.session_state.requirements, 1):
            st.write(f"{i}. Length: {req['length']}\", Width: {req['width']}\"")

        if st.button("Solve Cutting Pattern"):
            has_stock, missing_pieces = check_stock_availability(
                st.session_state.requirements,
                stock_manager.get_stock_for_processing()
            )

            if not has_stock:
                st.error("Insufficient stock! You need to purchase wood of at least these dimensions:")
                for piece in missing_pieces:
                    st.write(f"- Length: {piece['length']}\", Width: {piece['width']}\"")
            else:
                solution, remaining_pieces = solve_cutting_stock(
                    st.session_state.requirements,
                    stock_manager.get_stock_for_processing()
                )

                if solution and not remaining_pieces:
                    st.session_state.current_solution = solution
                    st.write("Proposed Cutting Pattern:")
                    for stock_id, cuts in solution:
                        stock_piece = stock_manager.get_stock_for_processing()[
                            stock_manager.get_stock_for_processing()['id'] == stock_id
                            ].iloc[0]
                        st.write(
                            f"\nFrom stock piece (length: {stock_piece['remaining_length']:.2f}\", width: {stock_piece['width']:.2f}\"):")
                        for cut in cuts:
                            width_note = " (needs width cut)" if cut['width_cut_needed'] else ""
                            st.write(f"- Cut piece: {cut['length']:.2f}\" × {cut['width']:.2f}\"{width_note}")

                    if st.button("Accept Solution"):
                        stock_manager.update_stock(solution)
                        st.session_state.requirements = []
                        st.rerun()
                else:
                    st.error(f"Cannot fit all pieces with current stock. Remaining pieces: {remaining_pieces}")

    if st.button("Clear Requirements"):
        st.session_state.requirements = []
        st.rerun()


if __name__ == "__main__":
    main()
