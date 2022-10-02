<html><head><meta content="text/html; charset=UTF-8" http-equiv="content-type"></head><body><div class="c40"><h1 class="c38" id="h.v0rgan1jjm8x"><span class="c15 c22 c34">Heart Rate Detection Using Remote Photoplethysmography</span></h1><p class="c4"><span class="c15 c22 c42">David Haas, Spencer Mullinix, Hogan Pope</span></p><p class="c4"><span class="c22 c33 c39">Fall 2020 ECE 4554/5554 Computer Vision: Course Project</span></p><p class="c4"><span class="c22 c33">Virginia Tech</span><hr></p><h3 class="c11" id="h.itx3moozawen"><span class="c14">Abstract</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 228.50px; height: 198.36px;"><img alt="" src="images/image5.png" style="width: 228.50px; height: 198.36px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h3><p class="c4"><span class="c1">Currently heart rate is an attribute that can be incredibly difficult to measure without being in close proximity to the patient. However, by using modern Computer Vision techniques, a close approximation to heart rate can be discovered with nothing more than a live video feed. This paper details a remote photoplethysmography implementation utilizing Fourier techniques and Independent Component Analysis to estimate a subject&rsquo;s BVP signal, and furthermore their heart rate. We report moderate success, with an root mean square error of 6.76. Further work should explore real-time implementations of our algorithm, along with reducing the use of priors within our work.</span></p><p class="c3"><span class="c1"></span></p><p class="c2 c17"><span class="c1"></span></p><p class="c3"><span class="c1"></span></p><h3 class="c11" id="h.xrf0xh2mhh1x"><span class="c15 c22 c27">Introduction</span></h3><p class="c4"><span class="c1">Finding a client&#39;s heart rate either for health, polygraph, or other reasons, is an issue that classically requires close proximity. However, with recent developments in remote photoplethysmography, this is no longer necessarily the case. If a stable version of this were deployed, it would allow for better remote healthcare work and improvements in other areas where being remote can help lower costs or increase availability. Our work only necessitates an RGB camera capable of recording video, as part of our goal is to make this capability available to as wide an array of people as possible. Hopefully being able to make them deployable on nearly all modern laptops, as well as potentially smartphones. One of the ways this issue has been approached in the past, specifically in the realm of smart phones, is through the use of fingerprint scanners. However, one of the benefits of being able to do this entirely via camera, is that while fingerprint scanners are becoming increasingly common in smartphones, they are all but non-existent in laptops, and many other devices that already have integrated cameras. Thus using only a camera would increase the domain of devices that could be supported.</span></p><p class="c3"><span class="c1"></span></p><h3 class="c11" id="h.7d7oyvfa4dm0"><span class="c15 c22 c27">Approach</span></h3><p class="c4"><span class="c1">We implemented an image processing pipeline aimed towards extracting a subject&#39;s blood volume pulse (BVP) signal, and from that, their pulse rate with a technique called remote photoplethysmography (rPPG). The algorithms are fed a video of a subject, and processed each frame of the video to extract time-indexed RGB vectors. The vectors then go through a pipeline of spectral and statistical analysis algorithms to extract a BVP signal, and from that, their heart rate.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c12">The spectral method we have implemented is inspired by Poe et al. [1] and consists of roughly three portions: ROI detection, preprocessing and extraction, and pulse rate calculation. The first of which is aimed to calculate the location of the subject&#39;s face to measure the BVP signal. To maintain a robust sequence of measurements on a relatively static portion of the subject&rsquo;s face, we used the DLib facial landmark detector to extract the area between the subject&rsquo;s cheeks [2], illustrated in </span><span class="c12 c30">Figure 6</span><span class="c12">. After their face has been segmented, each RGB channel in the face-image is averaged, resulting in one measurement per channel per frame in the video, producing three signals. These signals are illustrated in </span><span class="c12 c30">Figure 1</span><span class="c1">. </span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c12">As heart rate signals are non-stationary, we then detrended these signals using a smoothness priors approach (</span><span class="c12 c30">Figure 2</span><span class="c1">) with a cutoff frequency of 0.33 Hz [3]. After the RGB signals have been detrended and z-normalized, we use Independent Component Analysis (ICA) to decompose them into three independent source signals, shown in figure 3. To ensure that we could perform this step robustly and quickly, we opted to use scitkit-learn&rsquo;s FastICA implementation [4]. ICA separates color variations due to BVP from variations caused by motion, lighting, or other sources. One of the returned components represents the fluctuations in color caused by variations in blood volume; this is assumed to be the component with the largest peak in its power spectrum. An example of an extracted BVP signal is shown in figure 4.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c1">We then filter the signal in the time and frequency domains with a 5 point moving-average filter and a hamming window bandpass filter with cut-off frequencies depending on the user&rsquo;s inputted state. We allow them to choose between resting, recovery, and active; each of which have different cut-off frequencies that incorporate prior estimates of their heart rate. Once the BVP signal is calculated, we use the interbeat-interval estimation implementation described in van Gent et al [5] to estimate the heart rate of the subject. We decided to utilize these authors&rsquo; implementations because the core focus of our project is on remotely estimating the BVP signal, not interbeat-interval estimation.</span></p><p class="c3"><span class="c1"></span></p><p class="c3"><span class="c1"></span></p><p class="c3"><span class="c1"></span></p><h3 class="c11" id="h.wrh8w477mucc"><span class="c16 c22 c27">Experiments</span></h3><p class="c3"><span class="c8"></span></p><h4 class="c28" id="h.hrzisetzxl4d"><span class="c15 c22 c23">Experimental Methodology</span></h4><p class="c4"><span class="c1">Our testing approach consists of two phases of testing. The first phase consists of preliminary tests that test the initial operating capability of our project. Phase one tests the algorithm on data of people of the same skin tone, gender, with no facial hair, under ideal lighting conditions, with the test subject facing the camera straight on from a fixed distance. This methodology of testing will allow students to determine the algorithms operating capacity before other variables are introduced. Ideally, the second phase of testing would have introduced the variables that were fixed in the first phase. Each variable would have been changed independently of the rest to isolate that particular variable. This stage was unable to be completed due to lack of diverse data. The subjects available to us, for reasons that are explained in the data set section, were all white males between the ages of 18-25. One variable that was able to be added was facial hair.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c1">This methodology for testing the algorithm would have allowed for complete testing of variables while isolating the faults. The algorithm will be considered successful if the algorithm can achieve &plusmn;5% in phase one.</span></p><p class="c3"><span class="c1"></span></p><h4 class="c28" id="h.jspkfics25bl"><span class="c15 c23 c22">Data Sets</span></h4><p class="c4"><span class="c31"><a class="c10" href="https://www.google.com/url?q=https://mahnob-db.eu/hci-tagging/&amp;sa=D&amp;ust=1605924449853000&amp;usg=AOvVaw2otoEVeXGkXObdtovgovgN">HCI tagging database</a></span><span class="c1">: This database is videos and images of 30 test subjects&#39; biometric reactions to stimuli. This dataset includes images and biometric data for these subjects.</span></p><p class="c4"><span class="c31"><a class="c10" href="https://www.google.com/url?q=https://osf.io/fdrbh/wiki/home/&amp;sa=D&amp;ust=1605924449854000&amp;usg=AOvVaw3iL59UfplD1ue5Q_sh8iln">OSF rPPG</a></span><span class="c12">: This dataset covers provide RGB images and videos that are tagged with the foreground and background of the image as well as the biometrics of the people in the image. </span><span class="c31"><a class="c10" href="https://www.google.com/url?q=https://www.idiap.ch/dataset/cohface&amp;sa=D&amp;ust=1605924449854000&amp;usg=AOvVaw2SGXCnlUYxvl4UvJS4vjop">COHFACE dataset</a></span><span class="c1">: This dataset consists of 160 minutes of 40 individuals of varying genders with tagged biometric data. The only downside to this database is that we will need assistance gaining access. After discovering the </span></p><p class="c4"><span class="c1">Collected Data: Eleven videos were taken of ten different people in ten different lighting conditions. Their heart-rates were taken using established ppg algorithms. These readings were used as ground truth for training.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c1">Sadly, after further processing of the HCI data, including the unpackaging of EEG data, we discovered that while this database has a lot of useful biological information about patients, along with videos, it does not have the patient&#39;s heart rate, rendering this dataset useless. The other two major databases proved to be inaccessible given our lack of credentials as undergraduate students. So we had to rely entirely on the data we were able to collect ourselves. While useful, the PPG heart rate measurements we took of our subjects were only accurate to within two BPM, leading to the potential for significant error within our data, as well as the danger of overfitting due to lack of subjects. Additionally, due to covid restrictions in the time window we had to complete this project, we were unable to reach a diverse field of subjects to test our methods on, limiting us to white males between the ages of 18 and 25.</span></p><p class="c3"><span class="c1"></span></p><h4 class="c28 c44" id="h.xod1gz9jovmz"><span class="c15 c23 c22"></span></h4><h4 class="c28" id="h.hzudmkjon5s"><span class="c15 c23 c22">Phase 1</span></h4><p class="c4"><span>Shown in figure 1 is the mapping of our heart rate measurements to the ground truth values and the trend between them. Ideally, the trendline will follow y = x.</span></p><p class="c2"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 555.50px; height: 384.95px;"><img alt="" src="images/image4.png" style="width: 555.50px; height: 384.95px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c2"><span class="c1">Figure 1 - Experimental Results</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c1">Additionally, we compared our methods to other popular methods of RPPG(Remote Photoplethysmography) as seen in Figure 2 [6].</span></p><p class="c3"><span class="c1"></span></p><a id="t.5a322c3a684f517033695f73b5c4df0eed9c59c8"></a><a id="t.0"></a><table class="c32"><tbody><tr class="c37"><td class="c25" colspan="1" rowspan="1"><p class="c2 c17"><span class="c1"></span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c2"><span class="c12 c16">Standard Deviation</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c2"><span class="c12 c16">Mean Absolute Error</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c2"><span class="c12 c16">Root Mean Square Error</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c12 c15">Poh2011</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">13.5</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">13.6</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">CHROM</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">13.49</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">22.36</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">LI2014</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">6.88</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">7.62</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">SAMC</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">5.81</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">4.96</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">6.23</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">SynRhythm</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">10.88</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">11.08</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">HR-CNN</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">7.25</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">9.24</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">DeepPhys</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">4.57</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">-</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">rPPPGNet</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">7.82</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">5.51</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">7.82</span></p></td></tr><tr class="c18"><td class="c25" colspan="1" rowspan="1"><p class="c4"><span class="c15 c12">Our Method</span></p></td><td class="c13" colspan="1" rowspan="1"><p class="c4"><span class="c1">4.26</span></p></td><td class="c9" colspan="1" rowspan="1"><p class="c4"><span class="c1">5.25</span></p></td><td class="c26" colspan="1" rowspan="1"><p class="c4"><span class="c1">6.76</span></p><p class="c3"><span class="c1"></span></p></td></tr></tbody></table><h4 class="c41" id="h.jeilyf5a453g"><span class="c39 c22 c43">Figure 2 - Table of Our method and other common methods error rates</span></h4><p class="c4"><span class="c8">Compared to many other methods our method has outstanding results, however, it is important to note that due to limiting factors, our data set was substantially smaller than what was used to find these values, consisting of only 10 subjects. All under similar conditions. However, this does seem to still indicate the effectiveness of our methods.</span></p><p class="c4"><span class="c8">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></p><p class="c4"><span class="c8">The method we have presented has an error rate of approximately 7% which while higher than the goal of 5%, still indicates that this is a solid approach and potentially, that given more data, specifically, more accurate data, the approach shown could drop below 5% error.</span></p><p class="c3"><span class="c1"></span></p><p class="c3"><span class="c1"></span></p><h4 class="c28" id="h.58yaoe9kt2dg"><span class="c15 c23 c22">Special Test Cases</span></h4><p class="c4"><span>Test Case 1: Description</span></p><p class="c4"><span class="c12">We performed an experiment measuring the effect of subject distance from the camera on the accuracy of the derived PPG signal, compared to a reference measurement from a pulse oximeter sensor. We believe that distance may influence accuracy, due to decreasing pixel density of the face-image as the subject moves further from the camera. To test this, we will record 30 second videos of the subject 0.5 meters from the camera, 1 meter from the camera, and 2 meters from the camera. Variables such as camera location, lighting, and video quality will be held constant. We expect that the derived PPG signals of further away subjects will have a higher signal-to-noise ratio, and thus be less accurate.</span></p><p class="c4"><span class="c1">Test Case 1: Results</span></p><a id="t.f8bbf7ec675b82b33769932cd5a5b13045ee1851"></a><a id="t.1"></a><table class="c5"><tbody><tr class="c29"><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">Distance from Camera</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">0.5 M</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">1.0 M</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">2.0 M</span></p></td></tr><tr class="c29"><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">Mean Absolute Error</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">4.99</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">2.1</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">8.8</span></p></td></tr></tbody></table><p class="c4"><span class="c1">There doesn&rsquo;t appear to be much of a difference within 1 meter but there is a noticeable change outside of 1 meter.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c1">Test Case 2: Description</span></p><p class="c4"><span class="c12">An experiment will be performed on the image quality that is necessary to get valid data. One of the main flaws of image processing is the general lack of resistance to lower resolution images. One of the tests will downsampling the testing images and compare the accuracy. The other image quality test will be image resolution. This will involve downsampling the testing images to different degrees and taking metrics at each step. The expected performance in this situation is decreased accuracy. The goal is to determine how robust the algorithm is. This will be achieved by incrementally testing by downsampling and comparing the results.</span></p><p class="c4"><span class="c1">Test Case 2: Results</span></p><a id="t.96f940bcf7adfae09bef92bde1686e34420526da"></a><a id="t.2"></a><table class="c5"><tbody><tr class="c29"><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">Image Resolution</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">540p</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">720p</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">1080p</span></p></td></tr><tr class="c29"><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">Mean Absolute Error</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">4.5</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">2.6</span></p></td><td class="c6" colspan="1" rowspan="1"><p class="c0"><span class="c1">4.8</span></p></td></tr></tbody></table><p class="c4"><span class="c1">It appears that there is a sweet spot for accuracy at 720p where it is more effective than 540p and 1080p. This is unexpected. Possible reasons is that the downsampling removes some noise but maintains enough information. More testing would be needed to be determined</span></p><h3 class="c11 c36" id="h.j5q4k54vg0qk"><span class="c15 c22 c27"></span></h3><p class="c4"><span class="c23 c16 c22">Initial Experiments: Facial Recognition</span></p><p class="c4"><span class="c8">&nbsp; &nbsp; The project revolves around facial recognition, so it was imperative that this portion be prioritized. Currently, the software takes the input of a xml data from the HCI Tagging Dataset. This data is then parsed for faces using OpenCV. Then a bounding box is drawn around the center. The center of the face was decided to be used based off of previous projects in the same field as ours. The center of the face seemed to provide good data while limiting the outside factors. Once the bounding box is drawn the average RGB value for each color channel within the box is saved. This process is repeated for every frame of the video. Each color channel is saved off as a signal after the video has finished being processed as shown below.</span></p><p class="c3"><span class="c8"></span></p><p class="c4"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 473.33px;"><img alt="" src="images/image1.png" style="width: 624.00px; height: 473.33px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c2"><span class="c8">Figure 1 - Raw RGB Signals</span></p><p class="c2 c17"><span class="c8"></span></p><p class="c4"><span class="c23 c16 c22">Initial Experiments: Signal Filtering</span></p><p class="c3"><span class="c8"></span></p><p class="c2"><span class="c8">Detrended RGB signal data</span></p><h3 class="c11" id="h.uzr0qtlnguec"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 469.33px;"><img alt="" src="images/image2.png" style="width: 624.00px; height: 469.33px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h3><p class="c2"><span class="c8">Figure 2 - Detrended and normalized RGB signals</span></p><p class="c2 c17"><span class="c8"></span></p><p class="c4"><span>I</span><span>ndependent component analysis (ICA) was performed on the color channel signals. Independent component analysis is a method of taking input signals then breaking them down into the components that make up those signals. The data above labeled &ldquo;Detrended RGB signal data&rdquo; was input into an ICA algorithm and the output is shown below.</span></p><p class="c3"><span class="c8"></span></p><h3 class="c11" id="h.ktwkhy9vr2rl"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 468.00px;"><img alt="" src="images/image7.png" style="width: 624.00px; height: 468.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h3><p class="c2"><span class="c8">Figure 3 - ICA components</span></p><p class="c2 c17"><span class="c8"></span></p><p class="c4"><span>Below is an ICA component, which our algorithm has determined to be the most likely BVP signal. The signal chosen has the largest magnitude within our bandpass frequencies, discussed in the approach. </span></p><h3 class="c19" id="h.73pdyck9zoe0"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 220.00px;"><img alt="" src="images/image6.png" style="width: 624.00px; height: 220.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h3><p class="c2"><span class="c8">Figure 4 - The ICA component selected as our BVP Signal</span></p><p class="c2 c17"><span class="c8"></span></p><p class="c4"><span>After a signal is chosen from ICA, a bandpass filter is applied to eliminate high and low frequency noise. This can be done because a heart beat is typically between 0.7-4.0 Hz [1].</span></p><h3 class="c11" id="h.h4iv4bnw3ghg"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 432.00px;"><img alt="" src="images/image8.png" style="width: 624.00px; height: 432.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h3><p class="c2"><span>Figure 5 - Bandpass filtered BVP signal</span></p><p class="c3"><span class="c8"></span></p><p class="c4"><span class="c8">Once the bandpass filtered BVP signal is created, we implemented a library by the name of heartpy, which was designed for PPG type signals. It uses the root mean square of successive differences (RMSSD) along with the standard deviation of successive differences (SDSD) to find the heart rate hidden in a signal [7].</span></p><p class="c3"><span class="c8"></span></p><p class="c3"><span class="c8"></span></p><h3 class="c11" id="h.7qznsxyw9pii"><span class="c15 c22 c27">Qualitative results</span></h3><p class="c4"><span class="c1">We detail many of our results in the experiments section. However, there are a few results worth discussing in this section as well.</span></p><p class="c3"><span class="c1"></span></p><p class="c4"><span class="c12">Figure 6 shows a visualization we developed that runs while our algorithm extracts RGB data. The red box shows the ROI over which the average red, blue, and green signals are extracted from. We calculate this ROI </span><span class="c1">using the dlib library [8] for facial detection, we also use this library for predicting facial landmarks. We use landmarks identified along the jawline, and then shrink the region to give us an area below the eyes that&rsquo;s all skin. Which will hopefully eliminate as much noise as we can.</span></p><p class="c3"><span class="c1"></span></p><p class="c2"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 433.58px; height: 319.66px;"><img alt="" src="images/image3.png" style="width: 433.58px; height: 319.66px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c2"><span class="c1">Figure 6 - A screenshot of the algorithm extracting color data from the subject</span></p><p class="c2 c17"><span class="c1"></span></p><p class="c3"><span class="c8"></span></p><h3 class="c11" id="h.86hwqjfzdq5b"><span class="c15 c22 c27">Conclusion</span></h3><p class="c4"><span class="c8">This report has described an overview of our approach for remote photoplethysmography. We utilize a variety of signal processing and statistical techniques to remotely extract a heart rate signal from a patient. The basic pipeline is as follows: extract color signals from a face within a video, filter them and separate them into independent source signals, determine which source signal is the subject&rsquo;s BVP, and estimate their heart rate from that signal. To accomplish this, we wrote a program in Python that incorporates much of our own work mixed with supporting libraries from reputable authors.</span></p><p class="c3"><span class="c8"></span></p><p class="c4"><span class="c8">Although our implementation is complete, it is not perfect. Future work should investigate estimating heart rate without the use of priors about the subject&rsquo;s activity levels, as one&rsquo;s heart rate may not always fall within the bounds suggested by their activity level. Furthermore, our algorithm processes video at about 3 frames per second, so it would be difficult to implement it online. Real-time heart rate estimates are of much use to physicians and other concerned parties, so this would be a useful expansion of our work.</span></p><p class="c3"><span class="c8"></span></p><p class="c3"><span class="c1"></span></p><h3 class="c11" id="h.ot41mmqx0vp2"><span class="c15 c22 c27">Citations</span></h3><p class="c4"><span class="c1">1. Poh, M.-Z., McDuff, D.J., Picard, R.W.: Advancements in noncontact, multiparameter physiological measurements using a webcam. IEEE Trans. Biomed. Eng. 58, 7&ndash;11 (2011)</span></p><p class="c4"><span class="c12">2. </span><span class="c7 c22">Davis E. King. </span><span class="c12">Dlib-ml: A Machine Learning Toolkit</span><span class="c7 c22">. </span><span class="c12 c30">Journal of Machine Learning Research</span><span class="c7 c22">&nbsp;10, pp. 1755-1758, 2009</span></p><p class="c4"><span class="c1">3. M. P. Tarvainen, P. O. Ranta-Aho, and P. A. Karjalainen, &ldquo;An advanced detrending method with application to HRV analysis,&rdquo; IEEE Trans. Biomed. Eng., vol. 49, no. 2, pp. 172&ndash;175, Feb. 2002.</span></p><p class="c4"><span class="c12">4. </span><span class="c7 c22">Scikit-learn: Machine Learning in Python</span><span class="c7 c35">, Pedregosa </span><span class="c7 c30 c35">et al.</span><span class="c39 c7 c35">, JMLR 12, pp. 2825-2830, 2011.</span></p><p class="c4"><span class="c7 c35">5. </span><span class="c7">van Gent, P., Farah, H., van Nes, N. and van Arem, B., 2019. Analysing Noisy Driver Physiology Real-Time Using Off-the-Shelf Sensors: Heart Rate Analysis Software from the Taking the Fast Lane Project. </span><span class="c7 c30">Journal of Open Research Software</span><span class="c7">, 7(1), p.32. DOI: </span><span class="c7"><a class="c10" href="https://www.google.com/url?q=http://doi.org/10.5334/jors.241&amp;sa=D&amp;ust=1605924449886000&amp;usg=AOvVaw0K8RmWyukkeN8rAJbCpYTM">http://doi.org/10.5334/jors.241</a></span></p><p class="c4"><span class="c12">6</span><span class="c12">. Zitong Yu, Wei Peng, Xiaobai Li, Xiaopeng Hong, Guoying Zhao</span><span class="c12 c45">, </span><span class="c12">Center for Machine Vision and Signal Analysis, University of Oulu, Finland, MOE Key Lab. for Intelligent Networks and Network SecurityFaculty of Electronic and Information Engineering, Xi&rsquo;an Jiaotong University, PRC School of Information and Technology , Northwest University, PRC, &ldquo;Remote Heart Rate Measurement from Highly Compressed Facial Videos: anEnd-to-end Deep Learning Solution with Video Enhancement&rdquo;, </span><span class="c24"><a class="c10" href="https://www.google.com/url?q=https://arxiv.org/pdf/1907.11921.pdf&amp;sa=D&amp;ust=1605924449886000&amp;usg=AOvVaw0sAxY9JDNVX8UeW2lELd2k">https://arxiv.org/pdf/1907.11921.pdf</a></span></p><p class="c4"><span class="c1">7. van Gent, Paul &amp; Farah, Haneen &amp; Nes, Nicole &amp; Arem, B.. (2018). Heart Rate Analysis for Human Factors: Development and Validation of an Open Source Toolkit for Noisy Naturalistic Heart Rate Data.</span></p><p class="c4"><span class="c12">8. Davis E. King.</span><span class="c12"><a class="c10" href="https://www.google.com/url?q=http://jmlr.csail.mit.edu/papers/volume10/king09a/king09a.pdf&amp;sa=D&amp;ust=1605924449887000&amp;usg=AOvVaw2_r2BehTx47xN9i4OPInJi">&nbsp;</a></span><span class="c24"><a class="c10" href="https://www.google.com/url?q=http://jmlr.csail.mit.edu/papers/volume10/king09a/king09a.pdf&amp;sa=D&amp;ust=1605924449887000&amp;usg=AOvVaw2_r2BehTx47xN9i4OPInJi">Dlib-ml: A Machine Learning Toolkit</a></span><span class="c12">. </span><span class="c12 c30">Journal of Machine Learning Research</span><span class="c1">&nbsp;10, pp. 1755-1758, 2009</span></p><hr><p class="c3"><span class="c1"></span></p><p class="c20"><span class="c39 c22 c33">&copy; David Haas, Spencer Mullinix, Hogan Pope</span></p><p class="c3"><span class="c8"></span></p><p class="c3"><span class="c8"></span></p></div></body></html>
