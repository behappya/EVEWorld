try:
    import pkg_resources

    if not hasattr(pkg_resources, "packaging") and hasattr(pkg_resources, "extern"):
        pkg_resources.packaging = pkg_resources.extern.packaging
except Exception:
    pass
