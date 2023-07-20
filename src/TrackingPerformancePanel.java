import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.border.TitledBorder;

/**
 * Main panel for the GUI of the software for tracking performance evaluation
 *  
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 * 
 */

public class TrackingPerformancePanel extends JPanel
{
	private static final long serialVersionUID = -4532776597296338654L;

	ArrayList<TrackPair> trackPairs = new ArrayList<TrackPair>();
	ArrayList<TrackSegment> recoveredTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> correctTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> missedTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> spuriousTracks = new ArrayList<TrackSegment>();

	ArrayList<TrackSegment> referenceTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> candidateTracks = new ArrayList<TrackSegment>();

	JButton selectReferenceGroupsButton;
	JButton selectCandidateGroupsButton;

	MeasuresPanel measuresPanel;
	JButton pairButton = new JButton("Pair tracks");
	JFormattedTextField maxDistTF;
	NumberFormat maxDistFormat = NumberFormat.getNumberInstance();
	JLabel refGroupsLabel = new JLabel("No reference tracks loaded");
	JLabel candidateGroupsLabel = new JLabel("No candidate tracks loaded");

	ArrayList<Thread> runningThreads = new ArrayList<Thread>();

	public TrackingPerformancePanel()
	{
		setName("Tracking Performance Computation");

		this.setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.weightx = 1.0;
		c.weighty = 0;

		c.gridx = 0;
		c.gridy = 0;
		this.add(buildTrackGroupSelectionPanel(), c);

		c.gridx = 0;
		c.gridy = 1;
		this.add(buildPairingPanel(), c);

		measuresPanel = buildMeasuresPanel();
		c.gridx = 0;
		c.gridy = 2;
		this.add(measuresPanel, c);
	}

	/**
	 * @return the JPanel object that is used in the GUI for setting up the pairing options
	 * */
	private JPanel buildPairingPanel()
	{
		JPanel pairingPanel = new JPanel();
		pairingPanel.setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.weightx = 1.0;
		c.weighty = 0;

		c.gridx = 0;
		c.gridy = 0;
		pairingPanel.add(pairButton, c);
		pairButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent arg0) {
				pairTracks();
			}
		});

		JLabel maxDistLabel = new JLabel("Maximum distance between detections");
		c.gridx = 0;
		c.gridy = 1;
		pairingPanel.add(maxDistLabel, c);

		maxDistTF = new JFormattedTextField(maxDistFormat);
		maxDistTF.setText("5");
		c.gridx = 0;
		c.gridy = 2;
		pairingPanel.add(maxDistTF, c);
		pairingPanel.setBorder(new TitledBorder("Track pairing"));

		return pairingPanel;
	}

	/**
	 * @return the JPanel object that is used in the GUI for selecting the tracks to compare
	 * */
	private JPanel buildTrackGroupSelectionPanel()
	{
		JPanel groupSelectionPanel = new JPanel();
		groupSelectionPanel.setLayout(new GridBagLayout());
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.weightx = 1.0;
		c.weighty = 0;

		c.gridx = 0;
		c.gridy = 0;
		c.weightx = 0.20;
		selectReferenceGroupsButton = new JButton("Load Reference tracks");
		groupSelectionPanel.add(selectReferenceGroupsButton, c);
		selectReferenceGroupsButton.addActionListener(new ActionListener()
		{
			@Override
			public void actionPerformed(ActionEvent arg0) {
				selectReferenceGroup();
			}
		});

		c.gridx = 1;
		c.gridy = 0;
		c.weightx = 0.80;
		groupSelectionPanel.add(refGroupsLabel, c);

		c.gridx = 0;
		c.gridy = 1;
		c.weightx = 0.20;
		selectCandidateGroupsButton = new JButton("Load Candidate tracks");
		groupSelectionPanel.add(selectCandidateGroupsButton, c);
		selectCandidateGroupsButton.addActionListener(new ActionListener()
		{
			@Override
			public void actionPerformed(ActionEvent arg0) {
				selectCandidateGroup();
			}

		});

		c.gridx = 1;
		c.gridy = 1;
		groupSelectionPanel.add(candidateGroupsLabel, c);
		groupSelectionPanel.setBorder(new TitledBorder("Track groups selection"));
		return groupSelectionPanel;
	}

	final JCheckBox colorTracksBox = new JCheckBox();
	final JCheckBox displayPairsBox = new JCheckBox();
	final JCheckBox displaySpuriousBox = new JCheckBox();
	final JCheckBox displayMissingBox = new JCheckBox();

	/**
	 * Select the input file that corresponds to reference tracks
	 * */
	private void selectReferenceGroup() {
		referenceTracks.clear();
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setMultiSelectionEnabled(false);
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		fileChooser.setName("Reference tracks");
		int returnVal = fileChooser.showDialog(this, "Load");
		File file = null;
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			file = fileChooser.getSelectedFile();
		}
		if (file!=null)
		{
			ArrayList<TrackSegment> trackSegmentList1 = new ArrayList<TrackSegment>();
			try{
				trackSegmentList1.addAll(TrackExportAndImportUtilities.importTracksFile(file));
			}
			catch (Exception e)
			{
				JOptionPane.showMessageDialog(this.getParent(),
						"Error while loading reference tracks from file.\n Please specify a valid .xml file that\n is using the official tracking contest format.",
						"Loading error",
						JOptionPane.ERROR_MESSAGE);
				e.printStackTrace();
				return;
			}
			referenceTracks.addAll(trackSegmentList1);
			refGroupsLabel.setText(referenceTracks.size()+" tracks loaded from "+file.getName());
			return;
		}
		refGroupsLabel.setText("No reference tracks loaded");
	}

	/**
	 * Select the input file that corresponds to candidate tracks
	 * */
	private void selectCandidateGroup() {
		candidateTracks.clear();
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setMultiSelectionEnabled(false);
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		fileChooser.setName("Candidate tracks");
		int returnVal = fileChooser.showDialog(this, "Load");
		File file = null;
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			file = fileChooser.getSelectedFile();
		}
		if (file!=null)
		{
			ArrayList<TrackSegment> trackSegmentList1 = new ArrayList<TrackSegment>();
			try{
				trackSegmentList1.addAll(TrackExportAndImportUtilities.importTracksFile(file));
			}
			catch (Exception e)
			{
				JOptionPane.showMessageDialog(this.getParent(),
						"Error while loading candidate tracks from file.\n Please specify a valid .xml file that\n is using the official tracking contest format.",
						"Loading error",
						JOptionPane.ERROR_MESSAGE);
				e.printStackTrace();
				return;
			}
			candidateTracks.addAll(trackSegmentList1);
			candidateGroupsLabel.setText(candidateTracks.size()+" tracks loaded from "+file.getName());
			return;
		}
		candidateGroupsLabel.setText("No candidate tracks loaded");
	}

	/**
	 * Pair the reference and candidate tracks
	 * */
	private synchronized void pairTracks()
	{
		trackPairs.clear();
		measuresPanel.resetScores();

		Thread pairingThread = new Thread(){
			@Override
			public void run()
			{
				setGUIEnabled(false);
				try{
					if(!(referenceTracks.isEmpty() || candidateTracks.isEmpty()))
					{
						OneToOneMatcher matcher = new OneToOneMatcher(referenceTracks, candidateTracks);
						final DistanceTypes distType = DistanceTypes.DISTANCE_EUCLIDIAN;
						final double maxDist;
						try {
							maxDist = maxDistFormat.parse(maxDistTF.getText()).doubleValue();
						} catch (ParseException e) {
							JOptionPane.showMessageDialog(TrackingPerformancePanel.this.getParent(),
									"Please specify a positive number for the\n maximum distance between positions.",
									"Configuration error",
									JOptionPane.ERROR_MESSAGE);
							e.printStackTrace();
							return;
						}
						if (maxDist<0)
						{
							JOptionPane.showMessageDialog(TrackingPerformancePanel.this.getParent(),
									"Please specify a positive number for the\n maximum distance between positions.",
									"Configuration error",
									JOptionPane.ERROR_MESSAGE);
							return;
						}
						ArrayList<TrackPair> pairs = new ArrayList<TrackPair>();
						try {
							pairs.addAll(matcher.pairTracks(maxDist, distType));
						} catch (Exception e) {
							e.printStackTrace();
						}
						// remove spurious candidate tracks
						recoveredTracks.clear();
						correctTracks.clear();
						missedTracks.clear();
						spuriousTracks.clear();
						for (TrackPair tp:pairs)
						{
							if (tp.candidateTrack.getDetectionList().isEmpty())
							{
								tp.candidateTrack = null;
								missedTracks.add(tp.referenceTrack);
							}
							else
							{
								recoveredTracks.add(tp.referenceTrack);
								correctTracks.add(tp.candidateTrack);
							}
						}
						for (TrackSegment ts:candidateTracks)
						{
							if (!correctTracks.contains(ts))
								spuriousTracks.add(ts);
						}
						trackPairs.addAll(pairs);
						final PerformanceAnalyzer analyzer = new PerformanceAnalyzer(referenceTracks, candidateTracks, trackPairs);
						SwingUtilities.invokeLater(new Runnable() {
							@Override
							public void run() {
								measuresPanel.setScores(analyzer.getNumRefTracks(), analyzer.getNumCandidateTracks(), analyzer.getNumRefDetections(), analyzer.getNumCandidateDetections(), analyzer.getPairedTracksDistance(distType, maxDist), analyzer.getPairedTracksNormalizedDistance(distType, maxDist), analyzer.getFullTrackingScore(distType, maxDist), analyzer.getNumSpuriousTracks(), analyzer.getNumMissedTracks(), analyzer.getNumPairedTracks(), analyzer.getNumPairedDetections(maxDist), analyzer.getNumMissedDetections(maxDist), analyzer.getNumWrongDetections(maxDist));								
							}
						});
					}
				}
				finally
				{
					setGUIEnabled(true);
					runningThreads.remove(this);
				}
			}
		};
		runningThreads.add(pairingThread);
		pairingThread.start();
	}

	/**
	 * Enable/disable elements of the GUI with which the user can interact
	 * */
	private void setGUIEnabled(final boolean enabled)
	{
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				maxDistTF.setEnabled(enabled);
				pairButton.setEnabled(enabled);
				selectCandidateGroupsButton.setEnabled(enabled);
				selectReferenceGroupsButton.setEnabled(enabled);
				measuresPanel.saveResultsButton.setEnabled(enabled);
			}
		});
	}

	/**
	 * Panel that contains the tracking criteria information
	 * */
	class MeasuresPanel extends JPanel
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = -8802797303612586042L;

		JLabel pairsDistanceLabel = new JLabel();
		JLabel pairsNormalizedDistanceLabel = new JLabel();
		JLabel pairsFullDistanceLabel = new JLabel();

		JLabel spuriousTracksLabel = new JLabel();
		JLabel missedTracksLabel = new JLabel();
		JLabel correctTracksLabel = new JLabel();
		JLabel tracksSimilarityLabel = new JLabel();

		double pairsDistance;
		double pairsNormalizedDistance;
		double pairsFullDistance;

		int numSpuriousTracks;
		int numMissedTracks;
		int numCorrectTracks;
		int numRefTracks;
		int numCandidateTracks;
		double tracksSimilarity;

		JLabel numRecoveredDetectionsLabel = new JLabel();
		JLabel numMissedDetectionsLabel = new JLabel();
		JLabel numWrongDetectionsLabel = new JLabel();
		JLabel detectionsSimilarityLabel = new JLabel();


		int numRecoveredDetections;
		int numMissedDetections;
		int numWrongDetections;
		int numRefDetections;
		int numCandidateDetections;
		double detectionsSimilarity;

		JButton saveResultsButton;

		public MeasuresPanel()
		{
			this.setLayout(new GridBagLayout());
			GridBagConstraints c = new GridBagConstraints();
			c.fill = GridBagConstraints.HORIZONTAL;
			c.weightx = 1.0;
			c.weighty = 0;

			c.gridx = 0;
			c.gridy = 0;
			c.weightx = 1;
			c.gridwidth = 2;
			JPanel globalPanel = new JPanel();
			globalPanel.add(new JLabel("Global measures"));
			this.add(globalPanel, c);
			c.weightx = 0.25;
			c.gridwidth = 1;

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Pairing distance"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(pairsDistanceLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Normalized pairing score (alpha)"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(pairsNormalizedDistanceLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Full normalized score (beta)"), c);			

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(pairsFullDistanceLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 1;
			c.gridwidth = 2;
			JPanel tracksPanel = new JPanel();
			tracksPanel.add(new JLabel("Tracks"));
			this.add(tracksPanel, c);
			c.weightx = 0.25;
			c.gridwidth = 1;

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Similarity between tracks (Jaccard)"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(tracksSimilarityLabel, c);


			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of paired tracks"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(correctTracksLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of missed tracks"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(missedTracksLabel, c);


			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of spurious tracks"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(spuriousTracksLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 1;
			c.gridwidth = 2;
			JPanel detectionsPanel = new JPanel();
			detectionsPanel.add(new JLabel("Detections"));
			this.add(detectionsPanel, c);
			c.weightx = 0.25;
			c.gridwidth = 1;

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Similarity between detections (Jaccard)"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(detectionsSimilarityLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of paired detections"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(numRecoveredDetectionsLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of missed detections"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(numMissedDetectionsLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			this.add(new JLabel("Number of spurious detections"), c);

			c.gridx = 1;
			c.weightx = 0.75;
			this.add(numWrongDetectionsLabel, c);

			c.gridx = 0;
			c.gridy++;
			c.weightx = 0.25;
			c.gridwidth = 2;
			saveResultsButton = new JButton("Save results");
			saveResultsButton.addActionListener(new ActionListener(){
				@Override
				public void actionPerformed(ActionEvent arg0) {
					saveResults();
				}});
			this.add(saveResultsButton, c);			
			c.gridwidth = 1;

			resetScores();
		}

		/**
		 * update the GUI with the new values for the tracking criteria
		 * */
		public void setScores(int numRefTracks, int numCandidateTracks, int numRefDetections, int numCandidateDetections, double pairsDistance, double pairsNormalizedDistance, double pairsFullDistance, int numSpurious, int numMissed, int numCorrect,
				int numRecoveredDetections, int numMissedDetections, int numWrongDetections)
		{
			this.numCandidateDetections = numCandidateDetections;
			this.numCandidateTracks = numCandidateTracks;
			this.numRefDetections = numRefDetections;
			this.numRefTracks = numRefTracks;
			this.pairsDistance = pairsDistance;
			this.pairsNormalizedDistance = pairsNormalizedDistance;
			this.pairsFullDistance = pairsFullDistance;
			this.numSpuriousTracks = numSpurious;
			this.numMissedTracks = numMissed;
			this.numCorrectTracks = numCorrect;
			this.numRecoveredDetections = numRecoveredDetections;
			this.numMissedDetections = numMissedDetections;
			this.numWrongDetections = numWrongDetections;
			this.detectionsSimilarity = (double)numRecoveredDetections/((double)numRecoveredDetections + (double)numMissedDetections + (double)numWrongDetections);
			this.tracksSimilarity = (double)numCorrectTracks/((double)numCorrectTracks + (double) numMissedTracks + (double) numSpuriousTracks);
			SwingUtilities.invokeLater(
					new Runnable() {

						@Override
						public void run() {
							refreshLabels();	
						}
					}
					);
		}

		/**
		 * reset the tracking criteria and the GUI accordingly
		 * */
		public void resetScores()
		{
			this.pairsDistance = 0;
			this.pairsNormalizedDistance = 0;
			this.pairsFullDistance = 0;
			this.numSpuriousTracks = 0;
			this.numMissedTracks = 0;
			this.numCorrectTracks = 0;
			this.tracksSimilarity = 0;
			this.numCandidateDetections = 0;
			this.numCandidateTracks = 0;
			this.numRefTracks = 0;
			this.numRefDetections = 0;
			pairsDistanceLabel.setText(": -");
			pairsNormalizedDistanceLabel.setText(": -");
			pairsFullDistanceLabel.setText(": -");
			correctTracksLabel.setText(": -");
			missedTracksLabel.setText(": -");
			spuriousTracksLabel.setText(": -");
			tracksSimilarityLabel.setText(": -");

			this.numRecoveredDetections = 0;
			this.numMissedDetections = 0;
			this.numWrongDetections = 0;
			this.detectionsSimilarity = 0;
			numRecoveredDetectionsLabel.setText(": -");
			numMissedDetectionsLabel.setText(": -");
			numWrongDetectionsLabel.setText(": -");
			detectionsSimilarityLabel.setText(": -");
		}

		/**
		 * Refresh the GUI with respect to the values of the tracking criteria 
		 * */
		public void refreshLabels()
		{
			pairsDistanceLabel.setText(": "+pairsDistance);
			pairsNormalizedDistanceLabel.setText(": "+pairsNormalizedDistance);
			pairsFullDistanceLabel.setText(": "+pairsFullDistance);
			tracksSimilarityLabel.setText(": "+tracksSimilarity);
			correctTracksLabel.setText(": "+numCorrectTracks+" (out of "+numRefTracks+")");
			missedTracksLabel.setText(": "+numMissedTracks+" (out of "+numRefTracks+")");
			spuriousTracksLabel.setText(": "+numSpuriousTracks);
			//spuriousTracksLabel.setText(": "+numSpuriousTracks+" (out of "+numCandidateTracks+")");
			numRecoveredDetectionsLabel.setText(": "+numRecoveredDetections+" (out of "+numRefDetections+")");
			numMissedDetectionsLabel.setText(": "+numMissedDetections+" (out of "+numRefDetections+")");
			numWrongDetectionsLabel.setText(": "+numWrongDetections);
			//numWrongDetectionsLabel.setText(": "+numWrongDetections+" (out of "+numCandidateDetections+")");
			detectionsSimilarityLabel.setText(": "+detectionsSimilarity);
		}

		/**
		 * Save the tracking criteria in a text file
		 * */
		private void saveResults()
		{
			File file = getValidSaveFile();
			if (file == null)
				return;
			try {
				FileWriter outFile = new FileWriter(file);
				PrintWriter out = new PrintWriter(outFile);

				out.println(pairsDistance+"\t : pairing distance");
				out.println(pairsNormalizedDistance+"\t : normalized pairing score (alpha)");
				out.println(pairsFullDistance+"\t : full normalized score (beta)");
				out.println(numRefTracks+"\t : number of reference tracks");
				out.println(numCandidateTracks+"\t : number of candidate tracks");
				out.println(tracksSimilarity+"\t : Similarity between tracks (Jaccard)");
				//out.println(numCorrectTracks+"\t : number of paired tracks");
				out.println(numCorrectTracks+"\t : number of paired tracks (out of "+numRefTracks+")");
				out.println(numMissedTracks+"\t : number of missed tracks (out of "+numRefTracks+")");
				out.println(numSpuriousTracks+"\t : number of spurious tracks)");
				//out.println(numSpuriousTracks+"\t : number of spurious tracks (out of "+numCandidateTracks+")");
				out.println(numRefDetections+"\t : number of reference detections");
				out.println(numCandidateDetections+"\t : number of candidate detections");
				out.println(detectionsSimilarity+"\t : Similarity between detections (Jaccard)");
				//out.println(numRecoveredDetections+"\t : number of paired detections");
				out.println(numRecoveredDetections+"\t : number of paired detections (out of "+numRefDetections+")");
				out.println(numMissedDetections+"\t : number of missed detections (out of "+numRefDetections+")");
				out.println(numWrongDetections+"\t : number of spurious detections");
				//out.println(numWrongDetections+"\t : number of spurious detections (out of "+numCandidateDetections+")");
				out.close();
			} catch (IOException e){
				e.printStackTrace();
				//new FailedAnnounceFrame("Writing the save file has failed.");
			}
		}
	}
	
	/**
	 * @return a MeasuresPanel object that is used for displaying tracking performance measures
	 * */
	private MeasuresPanel buildMeasuresPanel()
	{
		MeasuresPanel measuresPanel = new MeasuresPanel();	
		measuresPanel.setBorder(new TitledBorder("Tracking performance"));
		return measuresPanel;
	}

	/**
	 * Ask the user to choose a file in which to save results
	 * @return a valid File object in which to save results
	 * */
	private File getValidSaveFile()
	{
		boolean hasValidSaveFile = false;
		File file = null;
		while(!hasValidSaveFile)
		{
			JFileChooser fileChooser = new JFileChooser();
			fileChooser.setMultiSelectionEnabled(false);
			fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
			fileChooser.setDialogTitle("Text file for saving results");
			int returnVal = fileChooser.showDialog(this, "Set as save file");
			if (returnVal == JFileChooser.APPROVE_OPTION) {
				file = fileChooser.getSelectedFile();
				if (file.exists())
				{
					int n = JOptionPane.showConfirmDialog(
							TrackingPerformancePanel.this.getParent(),
							"This file already exists. Do you want to overwrite it?",
							"Save tracking performance results",
							JOptionPane.YES_NO_CANCEL_OPTION);
					switch (n)
					{
					case JOptionPane.YES_OPTION:
						hasValidSaveFile = true;
						break;
					case JOptionPane.CANCEL_OPTION:
						hasValidSaveFile = true;
						file = null;
						break;
					case JOptionPane.NO_OPTION:
						hasValidSaveFile = false;
					}
				}
				else
					hasValidSaveFile = true;
			}
			else return null;
		}
		return file;
	}
}
