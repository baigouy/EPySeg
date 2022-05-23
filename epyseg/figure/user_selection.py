# below is the java code of my selection --> I should inspire myself from it
# see the best way todo that
# or maybe allow to handle only elements of a given scope --> in a way that simplifies a lot the things !!! --> think about it
# if scope is images --> only offer images options, if scope is panel --> offer panel options
# if scope is Figure --> only offer figure options


# package GUIs;
#
# import MyShapes.Figure;
# import MyShapes.Montage;
# import MyShapes.MyImage2D;
# import MyShapes.PARoi;
# import MyShapes.Row;
# import MyShapes.MyTextRow;
# import java.awt.geom.Point2D;
# import java.util.ArrayList;
#
# /**
#  * This class deals with selections
#  *
#  * @author benoit aigouy
#  */
# public class OptimalSelection {
#
#     public static Object getSelectionAt(ArrayList<Object> ROIs, boolean goDeeper, Point2D.Double mouseClick, Object objectsAlreadySelected) {
#         Object sel = null;
#         if (ROIs == null || ROIs.isEmpty()) {
#             return null;
#         }
#         int size = ROIs.size();
#         /* we loop in reverse order */
#         loop1:
#         for (int i = size - 1; i >= 0; i--) {
#             Object object = ROIs.get(i);
#             if (object == null) {
#                 continue;
#             }
#             if (objectsAlreadySelected instanceof MyImage2D) {
#                 /* we are at the lowest level --> we go up again */
#                 if ((object instanceof Figure) || (object instanceof Row) || (object instanceof Montage)) {
#                     if (object instanceof PARoi) {
#                         if (((PARoi) object).contains(mouseClick)) {
#                             return object;
#                         }
#                     }
#                 }
#             }
#             /* go deeper --> if selected object is a figure --> return the row under the mouse */
#             if (objectsAlreadySelected instanceof Figure) {
#                 Row montageContainingClick = ((Figure) objectsAlreadySelected).getRowAtPosition(mouseClick);
#                 if (montageContainingClick instanceof Row) {
#                     return montageContainingClick;
#                 }
#             }
#             /* go deeper --> if selected object is a row --> return the montage under the mouse */
#             if (objectsAlreadySelected instanceof Row) {
#                 Montage montageContainingClick = ((Row) objectsAlreadySelected).getMontageAtPosition(mouseClick);
#                 if (montageContainingClick instanceof Montage) {
#                     return montageContainingClick;
#                 }
#             }
#             /* go deeper --> if selected object is a Montage --> return the image2D under the mouse */
#             if (objectsAlreadySelected instanceof Montage) {
#                 MyImage2D montageContainingClick = ((Montage) objectsAlreadySelected).getMyImage2DAtPosition(mouseClick);
#                 if (montageContainingClick instanceof MyImage2D) {
#                     return montageContainingClick;
#                 }
#             }
#             if (object instanceof PARoi) {
#                 if (((PARoi) object).contains(mouseClick)) {
#                     return object;
#                 }
#             }
#         }
#         return sel;
#     }
#
#     /**
#      * adds to the current selection the newly selected object
#      *
#      * @param ROIs
#      * @param breakOnFirstEncounter
#      * @param onlyAllowSameInstanceAsSelection
#      * @param mouseClick
#      * @param objectsAlreadySelected
#      * @return
#      */
#     public static Object addToSelection(ArrayList<Object> ROIs, boolean breakOnFirstEncounter, boolean onlyAllowSameInstanceAsSelection, Point2D.Double mouseClick, Object objectsAlreadySelected) {
#         Object newSel = getSelection(ROIs, breakOnFirstEncounter, onlyAllowSameInstanceAsSelection, mouseClick, objectsAlreadySelected);
# //        System.out.println("potentially newly selected object " + newSel + " " + objectsAlreadySelected);
#         /* if there is no cur sel return new sel as cur sel */
#         if (objectsAlreadySelected == null) {
#             return newSel;
#         }
#         if (newSel == null) {
#             return objectsAlreadySelected;
#         }
#         /* if sel is already a list of selected objects add the new sel to the list and return it */
#         if (objectsAlreadySelected instanceof ArrayList) {
#             ArrayList<Object> sel = ((ArrayList<Object>) objectsAlreadySelected);
#             if (!sel.contains(newSel)) {
#                 sel.add(newSel);
#             }
#             return sel;
#         } else {
#             /* there is already a selected object but we create a list and add all objects to it */
#             ArrayList<Object> sel = new ArrayList<Object>();
#             sel.add(objectsAlreadySelected);
#             if (newSel != objectsAlreadySelected) {
#                 sel.add(newSel);
#             }
#             return sel;
#         }
#     }
#
#     public static Object getSelection(ArrayList<Object> ROIs, boolean breakOnFirstEncounter, boolean onlyAllowSameInstanceAsSelection, Point2D.Double mouseClick, Object objectsAlreadySelected) {
#         if (ROIs == null || ROIs.isEmpty()) {
#             return null;
#         }
#         int size = ROIs.size();
#         ArrayList<Object> foundObjects = new ArrayList<Object>();
#         /* we loop in reverse order */
#         loop1:
#         for (int i = size - 1; i >= 0; i--) {
#             Object object = ROIs.get(i);
#             /* we prevent the object from being selected twice */
#             if (object == objectsAlreadySelected) {
#                 continue;
#             }
#             if (objectsAlreadySelected != null && (objectsAlreadySelected instanceof ArrayList)) {
#                 /* prevent from selecting the same object again */
#                 ArrayList<Object> curObjects = ((ArrayList<Object>) objectsAlreadySelected);
#                 for (Object foundObject : curObjects) {
#                     if (object == foundObject) {
#                         continue loop1;
#                     }
#                 }
#             }
#             if (object instanceof PARoi) {
#                 if (((PARoi) object).contains(mouseClick)) {
#                     if (objectsAlreadySelected == null) {
#                         return object;
#                     }
#                     if (onlyAllowSameInstanceAsSelection) {
#                         /* if they are of the same instance take it directly otherwise look inside */
#  /* here we just get objects of the same type */
#                         if ((object instanceof Figure) && ((objectsAlreadySelected instanceof Figure) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Figure))) {
#                             if (breakOnFirstEncounter) {
#                                 return object;
#                             } else {
#                                 foundObjects.add(object);
#                             }
#                         }
#                         /* prevent MyTextRow to be selected with Rows even if they are Rows */
#                         if (((object instanceof MyTextRow)) && (((objectsAlreadySelected instanceof MyTextRow)) || ((objectsAlreadySelected instanceof ArrayList) && (((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyTextRow)))) {
#                             if (breakOnFirstEncounter) {
#                                 return object;
#                             } else {
#                                 foundObjects.add(object);
#                             }
#                         } else if (((object instanceof Row) && !(object instanceof MyTextRow)) && (((objectsAlreadySelected instanceof Row) && !(objectsAlreadySelected instanceof MyTextRow)) || ((objectsAlreadySelected instanceof ArrayList) && (((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Row) && !(((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyTextRow)))) {
#                             if (breakOnFirstEncounter) {
#                                 return object;
#                             } else {
#                                 foundObjects.add(object);
#                             }
#                         }
#                         if ((object instanceof Montage) && ((objectsAlreadySelected instanceof Montage) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Montage))) {
#                             if (breakOnFirstEncounter) {
#                                 return object;
#                             } else {
#                                 foundObjects.add(object);
#                             }
#                         }
#                         if ((object instanceof MyImage2D) && ((objectsAlreadySelected instanceof MyImage2D) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyImage2D))) {
#                             if (breakOnFirstEncounter) {
#                                 return object;
#                             } else {
#                                 foundObjects.add(object);
#                             }
#                         }
#                         if ((object instanceof Figure) && ((objectsAlreadySelected instanceof MyTextRow) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyTextRow))) {
#                             Object innerSelectedObject = ((Figure) object).getRowAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof MyTextRow) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         } else if ((object instanceof Figure) && (((objectsAlreadySelected instanceof Row) && !(objectsAlreadySelected instanceof MyTextRow)) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Row && !(((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyTextRow)))) {
#                             Object innerSelectedObject = ((Figure) object).getRowAtPosition(mouseClick);
#                             if ((innerSelectedObject instanceof Row) && !(innerSelectedObject instanceof MyTextRow) ) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                         /* if instance of objects differ then take the compatible one when grouping */
#                         if ((object instanceof Figure) && ((objectsAlreadySelected instanceof Montage) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Montage))) {
#                             Object innerSelectedObject = ((Figure) object).getRowAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof Row) {
#                                 innerSelectedObject = ((Row) innerSelectedObject).getMontageAtPosition(mouseClick);
#                             }
#                             if (innerSelectedObject instanceof Montage) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                         /* if instance of objects differ then take the compatible one when grouping */
#                         if ((object instanceof Figure) && ((objectsAlreadySelected instanceof MyImage2D) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyImage2D))) {
#                             Object innerSelectedObject = ((Figure) object).getRowAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof Row) {
#                                 innerSelectedObject = ((Row) innerSelectedObject).getMontageAtPosition(mouseClick);
#                             }
#                             if (innerSelectedObject instanceof Montage) {
#                                 innerSelectedObject = ((Montage) innerSelectedObject).getMyImage2DAtPosition(mouseClick);
#                             }
#                             if (innerSelectedObject instanceof MyImage2D) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                         /* if instance of objects differ then take the compatible one when grouping */
#                         if ((object instanceof Row) && ((objectsAlreadySelected instanceof Montage) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof Montage))) {
#                             //get the montage at position
#                             Object innerSelectedObject = ((Row) object).getMontageAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof Montage) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                         /* if instance of objects differ then take the compatible one when grouping */
#                         if ((object instanceof Row) && ((objectsAlreadySelected instanceof MyImage2D) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyImage2D))) {
#                             Object innerSelectedObject = ((Row) object).getMontageAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof Montage) {
#                                 innerSelectedObject = ((Montage) innerSelectedObject).getMyImage2DAtPosition(mouseClick);
#                             }
#                             if (innerSelectedObject instanceof MyImage2D) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                         /* if instance of objects differ then take the compatible one when grouping */
#                         if ((object instanceof Montage) && ((objectsAlreadySelected instanceof MyImage2D) || ((objectsAlreadySelected instanceof ArrayList) && ((ArrayList<Object>) objectsAlreadySelected).get(0) instanceof MyImage2D))) {
#                             //get the montage at position
#                             Object innerSelectedObject = ((Montage) object).getMyImage2DAtPosition(mouseClick);
#                             if (innerSelectedObject instanceof MyImage2D) {
#                                 if (breakOnFirstEncounter) {
#                                     return innerSelectedObject;
#                                 } else {
#                                     foundObjects.add(innerSelectedObject);
#                                 }
#                             }
#                         }
#                     } else {
#                         return object;
#                     }
#                 }
#             }
#         }
#         if (!foundObjects.isEmpty()) {
#             return foundObjects;
#         }
#         return null;
#     }
#
#     public static void main(String[] args) {
#         System.exit(0);
#     }
#
# }
