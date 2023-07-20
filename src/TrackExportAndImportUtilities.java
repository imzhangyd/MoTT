import java.io.File;
import java.util.ArrayList;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 * Track import/export utilities.
 *  Input and output files are in the .xml format used for the ISBI'2012 Particle Tracking Challenge
 *  Tracks are of the type TrackSegment that is used by the TrackManager plugin
 *  
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 * @author Fabrice de Chaumont
 * @author Ihor Smal
 */

public class TrackExportAndImportUtilities {

	/**
	 * Load TrackSegment objects from a .xml file.
	 * @param inputFile an .xml file containing track information in the format used for the ISBI'2012 Particle tracking challenge
	 * @return a list of TrackSegment objects that corresponds to the loaded tracks
	 * 
	 * */
	public static ArrayList<TrackSegment> importTracksFile (File inputFile) throws IllegalArgumentException
	{
		ArrayList<TrackSegment> trackArrayList = new ArrayList<TrackSegment>();
		Document document = XMLUtil.loadDocument( inputFile, true );
		Element root = XMLUtil.getRootElement( document );
		if ( root == null )
		{
			throw new IllegalArgumentException( "can't find: <root> tag." );
		}
		Element trackingSet = XMLUtil.getSubElements( root , "TrackContestISBI2012" ).get( 0 );

		if ( trackingSet == null )
		{
			throw new IllegalArgumentException( "can't find: <root><TrackContestISBI2012> tag." ) ;		
		}

		ArrayList<Element> particleElementArrayList = XMLUtil.getSubElements( trackingSet , "particle" );

		for ( Element particleElement : particleElementArrayList )
		{
			ArrayList<Element> detectionElementArrayList =
					XMLUtil.getSubElements( particleElement , "detection" );
			TreeMap<Integer, Detection> detections = new TreeMap<Integer, Detection>();
			for ( Element detectionElement : detectionElementArrayList )	
			{
				int t = XMLUtil.getAttributeIntValue( detectionElement, "t" , -1 );
				if ( t < 0 )
					throw new IllegalArgumentException( "invalid t value: " + t ) ;				
				if (detections.containsKey(new Integer(t)))
					throw new IllegalArgumentException( "duplicated detection for a single track at time " + t ) ;				
				double x = XMLUtil.getAttributeDoubleValue( detectionElement, "x" , 0 );
				double y = XMLUtil.getAttributeDoubleValue( detectionElement, "y" , 0 );
				double z = XMLUtil.getAttributeDoubleValue( detectionElement, "z" , 0 );		
				Detection newDetection = new Detection(x, y, z, t);
				newDetection.setDetectionType(Detection.DetectionTypes.REAL);
				detections.put(new Integer(t), newDetection);
			}
			// add detections in chronological order and cap gaps with virtual detections
			if (!detections.isEmpty())
			{
				TrackSegment track = new TrackSegment();
				Detection lastDetection = null;
				for (Entry<Integer, Detection> e:detections.entrySet())
				{
					Detection detection = e.getValue();
					if (lastDetection!=null)
					{
						// cap hole with virtual detections
						if (detection.getT()>lastDetection.getT()+1)
						{
							int lastT = lastDetection.getT();
							int nextT = detection.getT();
							double lastX = lastDetection.getX();
							double lastY = lastDetection.getY();
							double lastZ = lastDetection.getZ();
							double nextX = detection.getX();
							double nextY = detection.getY();
							double nextZ = detection.getZ();
							double gapT = 1/((double)nextT-(double)lastT);
							for (int t = lastT+1; t < nextT; t++)
							{
								// linear interpolation
								Detection interpolatedDetection = new Detection(
										lastX + (double)(t-lastT)*(nextX-lastX)*gapT,
										lastY + (double)(t-lastT)*(nextY-lastY)*gapT,
										lastZ + (double)(t-lastT)*(nextZ-lastZ)*gapT,
										t
										);
								System.out.println(interpolatedDetection.getX()+" "+interpolatedDetection.getY()+" "+interpolatedDetection.getZ());
								interpolatedDetection.setDetectionType(Detection.DetectionTypes.VIRTUAL);
								track.addDetection(interpolatedDetection);
							}
						}
					}
					track.addDetection(detection);
					lastDetection = detection;
				}
				trackArrayList.add( track );
			}
		}
		System.out.println( trackArrayList.size() +" track(s) succesfuly loaded.");
		return trackArrayList;
	}


	/**
	 * Export TrackSegment objects to a .xml file.
	 * @param file output .xml file containing track information in the format used for the ISBI'2012 Particle tracking challenge
	 * @param tracks a list of TrackSegment objects that corresponds to the tracks to save
	 * 
	 * */
	public static void exportTracks (File file, ArrayList<TrackSegment> tracks) throws IllegalArgumentException {
		Document document = XMLUtil.createDocument( true );

		Element dataSetElement = document.createElement("TrackContestISBI2012");

		XMLUtil.getRootElement( document ).appendChild( dataSetElement );


		for ( TrackSegment particle: tracks )
		{
			Element particleElement =  document.createElement("particle");
			dataSetElement.appendChild(particleElement);

			for ( Detection detection : particle.getDetectionList() )
			{
				Element detectionElement =  document.createElement("detection");
				particleElement.appendChild( detectionElement );
				XMLUtil.setAttributeDoubleValue( detectionElement , "x" , roundDecimals2(detection.getX()) );
				XMLUtil.setAttributeDoubleValue( detectionElement , "y" , roundDecimals2(detection.getY()) );
				XMLUtil.setAttributeDoubleValue( detectionElement , "z" , roundDecimals2(detection.getZ()) );
				XMLUtil.setAttributeIntValue( detectionElement , "t" , detection.getT() );
			}

		}
		XMLUtil.saveDocument( document , file );
	}

	private static double roundDecimals2(double value) {
		value = value * 1000d;
		value = Math.round(value);
		return value/1000d;       
	}
}