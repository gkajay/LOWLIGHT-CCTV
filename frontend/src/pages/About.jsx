export default function About() {
  return (
    <div className="stack about">
      <section className="panel">
        <h2>What this does</h2>
        <p>
          CCTV cameras lose detail at night. This tool brightens dark frames with three deep-learning models and
          scores the result against a normal-light reference, so you can see which one recovers a scene better.
        </p>
      </section>

      <section className="panel">
        <h2>The three models</h2>
        <div className="grid-2">
          <article className="model-card cnn">
            <h3>Custom CNN</h3>
            <p>
              Four 3×3 convolution layers (3→32→32→16→3) with ReLU and a final sigmoid. It maps a dark image straight to
              a bright one. Trained on LOL image pairs with an L1 plus SSIM loss.
            </p>
          </article>
          <article className="model-card zdce">
            <h3>Zero-DCE++</h3>
            <p>
              A small network that predicts a brightness curve for every pixel, then applies that curve eight times.
              It learns from dark images alone, guided by exposure, colour and smoothness losses. It runs at a reduced
              resolution, so it is fast.
            </p>
          </article>
          <article className="model-card gan">
            <h3>GAN</h3>
            <p>
              The same network as the Custom CNN, used as a generator and fine-tuned against a second network, the
              discriminator, which tries to tell generated images from real bright photos. This pushes results toward
              a natural look. It is as fast as the CNN, but may score slightly lower on PSNR.
            </p>
          </article>
        </div>
      </section>

      <section className="panel">
        <h2>How to read the scores</h2>
        <dl>
          <dt>PSNR</dt>
          <dd>Pixel-level error against the reference, in decibels. Higher means closer.</dd>
          <dt>SSIM</dt>
          <dd>Similarity of structure, contrast and brightness, from 0 to 1. Closer to 1 means more alike.</dd>
          <dt>Time</dt>
          <dd>Milliseconds to enhance one image on this machine. It changes with image size and hardware.</dd>
        </dl>
      </section>

      <section className="panel">
        <h2>What it cannot do</h2>
        <ul>
          <li>Enhancement cannot recover detail the camera never captured, such as pure darkness or heavy motion blur.</li>
          <li>The models were trained and tested on LOL photographs, not real CCTV footage.</li>
          <li>It improves visibility only. It does not detect people, vehicles or events.</li>
          <li>Higher PSNR does not always look better. Check the images, not just the numbers.</li>
        </ul>
      </section>
    </div>
  );
}
