def int_to_argb_hex(value, has_alpha=False, auto_rint=True):
    if auto_rint and not isinstance(value, int):
        value = int(round(value))
    alpha = (value >> 24) & 0xFF
    red = (value >> 16) & 0xFF
    green = (value >> 8) & 0xFF
    blue = value & 0xFF

    if has_alpha:
        argb_hex = f"{alpha:02X}{red:02X}{green:02X}{blue:02X}"
    else:
        argb_hex = f"{red:02X}{green:02X}{blue:02X}"
    return argb_hex


if __name__ == '__main__':
    # Example usage:
    integer_value = 0xFFAABBCC  # Replace this with your integer value
    integer_value = 255  # Replace this with your integer value
    integer_value = 255.9  # Replace this with your integer value
    argb_hex_value = int_to_argb_hex(integer_value)
    print(f"ARGB Hex Value: #{argb_hex_value}")