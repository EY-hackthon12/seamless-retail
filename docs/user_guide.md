# User Guide: Running the Demo

## 1. Setup & Installation

### Option A: Docker (Recommended)
The easiest way to run the full system.
1.  Ensure Docker Desktop is running.
2.  Add your `OPENAI_API_KEY` to a `.env` file or export it.
3.  Run:
    ```bash
    docker-compose up --build
    ```
4.  Open `http://localhost:8000` to see the app.

### Option B: Local Development
If you want to modify code:
1.  **Backend**:
    ```bash
    pip install -r requirements.txt
    uvicorn app.main:app --reload
    ```
2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## 2. Demo Walkthrough

### Step 1: Mobile Engagement
1.  Navigate to the **Mobile App** view.
2.  Type a query like: *"I'm looking for a navy blue suit for a wedding."*
3.  The agent will respond, confirming availability (mocked as available).
4.  The system now "knows" you are interested in a blue suit.

### Step 2: The "Bridge"
1.  Click the **QR Code Icon** in the top-right corner of the Mobile Chat.
2.  A QR code representing your unique `session_id` will appear.

### Step 3: In-Store Experience
1.  Switch to the **Kiosk Dashboard** view.
2.  Click **"Scan with Camera"** (if you have a webcam) or **"Simulate Scan"** (for a quick demo).
3.  If using the camera, point it at the QR code on the mobile screen.
4.  **Magic Moment**: The Kiosk recognizes you!
    -   *Welcome Message*: "Welcome back!"
    -   *Recommendation*: "We found these Brown Leather Oxfords that go perfectly with the Blue Suit you were looking at."

## 3. Managing Local Models
The system includes a `ModelManager` to run your own models.
1.  Place your model files (GGUF, etc.) in the `models/` directory.
2.  Use the API to load them: `POST /api/models/load`.
3.  The agents can now be configured to use these models for inference.
