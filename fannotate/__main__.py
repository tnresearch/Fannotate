from .main import create_ui

if __name__ == "__main__":
    demo = create_ui()
    # Launch the interface with custom configurations
    demo.launch(
        server_name="0.0.0.0",  # Makes the server accessible from other machines
        server_port=1337,       # Default fannotate port
        share=False,            # Set to True to create a public link
        debug=True             # Enable debug mode for more detailed logs
    ) 